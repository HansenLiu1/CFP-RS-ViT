"""Unified logging system with timestamps, file output, and tqdm integration."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Redirect logging through tqdm to avoid progress bar interference."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logger(
    run_root: Path | str | None = None,
    name: str = "cfra",
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup logger with console + optional file output.
    
    Args:
        run_root: If provided, also log to <run_root>/logs/main.log
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False
    
    # Format: [2026-08-24 14:23:07] [cfra] [INFO] message
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (through tqdm)
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if run_root provided)
    if run_root is not None:
        log_dir = Path(run_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "main.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_stage_start(logger: logging.Logger, stage: str, step: int, total: int, details: str = "") -> None:
    """Log stage start with step counter."""
    msg = f"[Step{step:02d}/{total:02d}] [{stage}] 开始"
    if details:
        msg += f" - {details}"
    logger.info(msg)


def log_stage_complete(logger: logging.Logger, stage: str, elapsed_seconds: float, details: str = "") -> None:
    """Log stage completion with elapsed time."""
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        time_str = f"{hours}h{minutes:02d}m{seconds:02d}s"
    elif minutes > 0:
        time_str = f"{minutes}m{seconds:02d}s"
    else:
        time_str = f"{seconds}s"
    
    msg = f"[{stage}] 完成 (耗时: {time_str})"
    if details:
        msg += f" - {details}"
    logger.info(msg)


def create_progress_bar(
    total: int,
    desc: str,
    unit: str = "it",
    position: int = 0,
    leave: bool = True,
    **kwargs: Any,
) -> tqdm:
    """Create a tqdm progress bar with consistent styling.
    
    Args:
        total: Total iterations
        desc: Description text
        unit: Unit name
        position: Position for nested progress bars (0=outermost)
        leave: Whether to leave the bar on completion
        **kwargs: Additional tqdm arguments
        
    Returns:
        Configured tqdm instance
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        position=position,
        leave=leave,
        smoothing=0.05,  # Stable ETA
        ncols=120,
        file=sys.stderr,
        **kwargs,
    )
