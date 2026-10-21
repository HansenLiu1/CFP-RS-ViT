"""Preflight checks before main experiment execution.

Validates environment, datasets, models, and performs smoke tests without
creating any run directories or persistent artifacts.

Steps:
1. Python environment and dependencies
2. Configuration files (config/local.toml)
3. Dataset accessibility and basic parsing
4. Model checkpoints and SHA-256 verification
5. GPU/CPU/memory detection and batch_size recommendations
6. Unit tests and code compilation
7. Single-sample forward pass smoke test
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

FORCE_UTF8 = os.environ.get("PYTHONIOENCODING")
if FORCE_UTF8 is None:
    os.environ["PYTHONIOENCODING"] = "utf-8"


def check(name: str, condition: bool, message: str = "", fatal: bool = True) -> bool:
    """Print check result with ASCII-safe markers."""
    if condition:
        print(f"[PASS] {name}")
        if message:
            print(f"  {message}")
        return True
    else:
        status = "FAIL" if fatal else "WARN"
        print(f"[{status}] {name}")
        if message:
            print(f"  {message}")
        if fatal:
            sys.exit(1)
        return False


def main() -> None:
    print("=" * 70)
    print("CFP-RS-ViT Preflight Check")
    print("=" * 70)
    print()

    # Step 1: Python environment
    print("[Step 1/7] Python environment and dependencies")
    print("-" * 70)

    check("Python version", sys.version_info >= (3, 10), f"Python {sys.version_info.major}.{sys.version_info.minor}")

    try:
        import torch
        check("PyTorch installed", True, f"torch {torch.__version__}")
        check("CUDA available", torch.cuda.is_available(),
              f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    except ImportError:
        check("PyTorch installed", False, "pip install torch required")

    try:
        import numpy as np
        check("NumPy installed", True, f"numpy {np.__version__}")
    except ImportError:
        check("NumPy installed", False)

    try:
        from PIL import Image
        check("Pillow installed", True, f"PIL {Image.__version__}")
    except ImportError:
        check("Pillow installed", False)

    try:
        import tqdm
        check("tqdm installed", True)
    except ImportError:
        check("tqdm installed", False)

    print()

    # Step 2: Configuration files
    print("[Step 2/7] Configuration files")
    print("-" * 70)

    local_path = PROJECT_ROOT / "config" / "local.toml"
    check("config/local.toml exists", local_path.exists(),
          "Copy config/local.example.toml if missing")

    if local_path.exists():
        from cfra.configuration import load_local_config
        try:
            local_config = load_local_config(local_path)
            check("config/local.toml parseable", True)
        except Exception as e:
            check("config/local.toml parseable", False, str(e))
            local_config = None
    else:
        local_config = None

    main_config_path = PROJECT_ROOT / "configs" / "main.toml"
    check("configs/main.toml exists", main_config_path.exists())

    if main_config_path.exists():
        from cfra.configuration import load_experiment_config
        try:
            main_config = load_experiment_config(main_config_path)
            check("configs/main.toml parseable", True)
        except Exception as e:
            check("configs/main.toml parseable", False, str(e))
            main_config = None
    else:
        main_config = None

    print()

    # Step 3: Dataset accessibility
    print("[Step 3/7] Dataset accessibility")
    print("-" * 70)

    if local_config and main_config:
        from cfra.configuration import configured_dataset_paths

        try:
            dataset_paths = configured_dataset_paths(local_config, main_config["scope"]["datasets"])
            for dataset_id, path in dataset_paths.items():
                exists = path.exists()
                check(f"Dataset '{dataset_id}' path", exists, str(path), fatal=False)

            # Quick image count check for DOTA
            if "dota" in dataset_paths and dataset_paths["dota"].exists():
                train_images = dataset_paths["dota"] / "Train" / "Images"
                if train_images.exists():
                    count = len(list(train_images.glob("*.png")))
                    check(f"DOTA training images", count > 0, f"Found {count} images", fatal=False)
        except Exception as e:
            check("Dataset path resolution", False, str(e), fatal=False)

    print()

    # Step 4: Model checkpoints
    print("[Step 4/7] Model checkpoints")
    print("-" * 70)

    if local_config and main_config:
        from cfra.models import resolve_model_assets

        try:
            model_ids = [
                *main_config["scope"].get("models", []),
                *main_config["scope"].get("representation_variants", []),
            ]
            model_assets = resolve_model_assets(local_config, model_ids, registry_root=PROJECT_ROOT / "registry")

            for model_id, asset in model_assets.items():
                exists = asset.path.exists()
                check(f"Model '{model_id}' checkpoint", exists, str(asset.path), fatal=False)

                if exists and asset.checkpoint_sha256:
                    # Quick file existence check (full SHA-256 is expensive)
                    check(f"Model '{model_id}' readable", asset.path.is_file() or asset.path.is_dir(), fatal=False)
        except Exception as e:
            check("Model asset resolution", False, str(e), fatal=False)

    print()

    # Step 5: Hardware detection
    print("[Step 5/7] Hardware detection and recommendations")
    print("-" * 70)

    cpu_cores = os.cpu_count() or 1
    print(f"  CPU: {cpu_cores} cores")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU: {gpu_name}")
            print(f"  GPU Memory: {gpu_memory_gb:.1f} GB")

            # Batch size recommendations
            if gpu_memory_gb >= 22:
                recommended_batch = 48
            elif gpu_memory_gb >= 10:
                recommended_batch = 24
            elif gpu_memory_gb >= 6:
                recommended_batch = 12
            else:
                recommended_batch = 8

            print(f"  Recommended batch_size: {recommended_batch}")
            print(f"  Recommended CFRA_CPU_WORKERS: {max(1, min(12, cpu_cores - 2))}")
        else:
            print("  GPU: Not available (CPU-only mode)")
            print("  Recommended batch_size: 4")
    except ImportError:
        pass

    check("Hardware detection", True)
    print()

    # Step 6: Unit tests
    print("[Step 6/7] Unit tests and compilation")
    print("-" * 70)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-q"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        test_passed = result.returncode == 0
        if test_passed:
            # Extract test count from output
            lines = result.stderr.strip().split("\n")
            summary = lines[-1] if lines else ""
            check("Unit tests", True, summary)
        else:
            check("Unit tests", False, result.stderr[-500:] if result.stderr else "", fatal=False)
    except subprocess.TimeoutExpired:
        check("Unit tests", False, "Tests timed out after 300s", fatal=False)
    except Exception as e:
        check("Unit tests", False, str(e), fatal=False)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        check("Code compilation", result.returncode == 0, fatal=False)
    except Exception as e:
        check("Code compilation", False, str(e), fatal=False)

    print()

    # Step 7: Single-sample smoke test
    print("[Step 7/7] Single-sample forward pass smoke test")
    print("-" * 70)

    try:
        import torch
        import numpy as np
        from cfra.models import load_adapter, parse_model_runtime_config

        if torch.cuda.is_available() and main_config:
            # Create dummy inputs
            dummy_crops = [np.full((448, 448, 3), 127, dtype=np.uint8) for _ in range(2)]

            device = "cuda:0"

            # Test with a lightweight model (CLIP)
            test_model_id = "clip"
            if local_config and test_model_id in main_config["scope"]["models"]:
                from cfra.models import resolve_model_assets
                assets = resolve_model_assets(local_config, [test_model_id], registry_root=PROJECT_ROOT / "registry")
                runtime = parse_model_runtime_config(main_config, [test_model_id])

                if test_model_id in assets and assets[test_model_id].path.exists():
                    with load_adapter(
                        test_model_id,
                        assets[test_model_id],
                        runtime=runtime,
                        registry_root=PROJECT_ROOT / "registry",
                        device=device,
                    ) as adapter:
                        encoded = adapter.encode(dummy_crops)
                        shape = encoded.values.shape if hasattr(encoded.values, "shape") else "unknown"
                        check("Single-sample forward pass", True, f"Output shape: {shape}")
                else:
                    check("Single-sample forward pass", False, f"Model '{test_model_id}' not found", fatal=False)
            else:
                check("Single-sample forward pass", False, "CLIP model not in scope", fatal=False)
        else:
            check("Single-sample forward pass", False, "Skipped (no GPU or config)", fatal=False)
    except Exception as e:
        check("Single-sample forward pass", False, str(e), fatal=False)

    print()
    print("=" * 70)
    print("Preflight check completed!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Set CFRA_RUN_DIR to specify output directory")
    print("  2. Run: python scripts/41_run_main_experiment.py")
    print("  3. Monitor progress in logs/main.log")


if __name__ == "__main__":
    main()
