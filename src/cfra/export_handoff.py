"""Lightweight 50 -> 51 artifact contract; no combined metrics materialization."""
from pathlib import Path
import json
from datetime import datetime, timezone


def write_export_handoff(root, config):
    root = Path(root)
    scope = config['scope']
    expected = []
    def require(stage, relative):
        path = root / relative
        expected.append(dict(stage=stage, artifact=relative, present=path.is_file(),
                             bytes=path.stat().st_size if path.is_file() else None))
    for stage in scope['experiment_ids']:
        if stage == 'E04-5' and not config.get('experiment_controls', {}).get(stage, {}).get('enabled', False):
            continue
        if stage in ['E00', 'E01', 'E02']:
            require(stage, f'manifest/{stage.lower()}.json')
        elif stage == 'E03':
            # E03 is written per model by the current runner and as a single
            # whole-stage file by older runs.  Either layout is a complete
            # stage, so accept whichever exists instead of forcing a rerun.
            legacy = root / 'intermediates/stage_results/E03.json'
            if legacy.is_file():
                require(stage, 'intermediates/stage_results/E03.json')
            else:
                for model in scope['models']:
                    require(stage, f'intermediates/stage_results/E03_{model}.json')
        elif stage == 'E06':
            for model in scope['models']:
                for dataset in scope['datasets']:
                    require(stage, f'metrics/E06_{model}_{dataset}.json')
                    for suffix in ['json', 'npz']:
                        require(stage, f'intermediates/features/layerwise_{model}_{dataset}.{suffix}')
        elif stage == 'E08':
            for model in scope['models']:
                for dataset in scope['datasets']:
                    for pert in scope['perturbations']:
                        require(stage, f'metrics/E08_{model}_{dataset}_{pert}.json')
        else:
            for model in scope['models']:
                require(stage, f'intermediates/stage_results/{stage}_{model}.json')
    for name in ['run.json', 'models.csv', 'images.csv', 'objects.csv', 'splits.csv']:
        require('protocol', 'manifest/' + name)
    selected_models = set(scope['models']) | set(scope.get('representation_variants', []))
    for model in config.get('text_baseline_protocol', {}).get('models', []):
        if model in selected_models and 'E08' in scope['experiment_ids']:
            require('E08text', f'metrics/E08_text_{model}.json')
    missing = [r for r in expected if not r['present'] or not r['bytes']]
    result = dict(schema_version=1, run_id=root.name, split_seed=config['split']['seed'],
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  status='ready_for_export_validation' if not missing else 'incomplete',
                  scientific_validation='not_performed_by_handoff',
                  artifacts=expected, missing=missing,
                  next_command='python scripts/51_export_paper_data.py')
    (root/'manifest/export_handoff.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
