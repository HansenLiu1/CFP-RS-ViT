"""Provenance-first long-format export. Missing evidence is never imputed.

Stage artifacts are authoritative; old paper CSVs and combined core_results
are deliberately not read (they can survive stage invalidation).
"""
from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import itertools
import json
import math
import os
import pickle
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import ClassVar

import numpy as np

TABLES = [
    'population', 'split_roles', 'split_overlap', 'legacy_split', 'specificity_full',
    'specificity_classwise', 'baseline_comparison', 'prototype_cfp_complementarity',
    'transfer_perturbation', 'raw_roi_transfer', 'retrieval', 'layerwise_specificity',
    'layerwise_alignment', 'low_rank', 'common_mode_stability', 'strength_sweep',
    'normalization', 'support_scaling', 'context_factorial', 'seed_stability',
    'removal_dose', 'removal_controls', 'readout_refit', 'model_registry',
    'prompt_text_baseline', 'statistical_summary', 'main_text_numbers',
    'claim_registry', 'artifact_registry', 'layerwise_readout',
]
FIELDS = ['result_id', 'dataset', 'model', 'checkpoint', 'layer', 'space', 'perturbation', 'perturbation_strength', 'operator', 'normalization', 'support_size', 'direction_fit_role', 'readout_fit_role', 'evaluation_role', 'split_seed', 'metric', 'estimate', 'CI_low', 'CI_high', 'valid_n_objects', 'n_source_images', 'p_value', 'FDR_q', 'artifact', 'run_id', 'stage', 'control', 'analysis', 'source_perturbation', 'target_perturbation', 'query_feature_space', 'gallery_feature_space', 'source_direction_space', 'condition', 'alpha', 'readout', 'class_id', 'source_image', 'mean_abs_cos_h_w', 'align_prototype', 'align_linear_weight', 'bootstrap_valid_replicates', 'bootstrap_replicates', 'permutation_replicates', 'statistics_status', 'status', 'missing_reason', 'aggregation', 'family']
METRICS = ['own_argmax_rate', 'mean_diagonal_margin', 'mean_own_cosine', 'macro_ovr_auc',
           'multiclass_accuracy', 'macro_ap', 'ndcg_at_k', 'precision_at_k', 'recall_at_k',
           'background_interaction_mean_l2', 'clean_background_delta_mean_l2',
           'degraded_background_delta_mean_l2', 'clean_degraded_delta_cosine_mean',
           'clean_degraded_class_mean_delta_cosine']
ROLES = ['direction_fit', 'readout_fit', 'validation', 'confirmation']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value):
    return isinstance(value, (float, int, np.number)) and math.isfinite(value)


def cell(value):
    if value is None or isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return 'NaN'
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=True)
    return value


def write_csv(path, rows, fields=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys([*fields, *(k for row in rows for k in row)]))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: cell(row.get(k)) for k in columns})


def default_progress(message):
    print(message, flush=True)


class _UnitSink:
    """File-like sink that streams source-unit rows into the active work unit.

    ``PaperExport.units`` used to be a gzip handle opened directly on the final
    ``source_units.jsonl.gz``.  It now writes into a per-work-unit gzip file, so
    an interrupted export never leaves a partially written final artifact and a
    resumed run can rebuild that file deterministically from committed units.
    """

    __slots__ = ('_owner',)

    def __init__(self, owner):
        self._owner = owner

    def write(self, text):
        self._owner._write_unit_line(text)

    def close(self):
        pass


class PaperExport:
    def __init__(self, runs, output, *, bootstrap_replicates=5000, permutation_replicates=10000,
                 recompute_layers=True, recompute_final=True, progress=None, bootstrap_workers=None,
                 exclude_stages=(), clear_checkpoint=False):
        self.runs = [Path(p).resolve() for p in runs]
        self.output = Path(output).resolve()
        self.exclude_stages = tuple(exclude_stages)
        if any(self.output == r or self.output in r.parents for r in self.runs):
            raise ValueError('Output must not replace a run or its parent')
        self.tables = defaultdict(list)
        self.master = []
        self.artifacts = {}
        self.bootstrap_replicates = bootstrap_replicates
        self.permutation_replicates = permutation_replicates
        self.recompute_layers = recompute_layers
        self.recompute_final = recompute_final
        if bootstrap_workers is None:
            env_value = os.environ.get('CFRA_BOOTSTRAP_WORKERS')
            bootstrap_workers = env_value if env_value else max(1, min((os.cpu_count() or 4) - 1, 18))
        self.bootstrap_workers = max(1, int(bootstrap_workers))
        self.next_result = 0
        self._started = time.time()
        self._progress = progress or default_progress
        self.output.mkdir(parents=True, exist_ok=True)
        # Committed work units make the export resumable after an interruption.
        # ``_checkpoint/units/<seq>.pkl`` stores the rows a unit added (and the
        # artifacts it touched); ``<seq>.units.jsonl.gz`` stores its source-unit
        # stream. ``_checkpoint/meta.json`` is the commit point.
        self._ckpt_dir = self.output/'_checkpoint'
        self._units_dir = self._ckpt_dir/'units'
        self._meta_path = self._ckpt_dir/'meta.json'
        self._meta = {'version': 1, 'seq': 0, 'next_result': 0}
        self._done = set()
        self._unit = None
        self._unit_snapshot = None
        self._unit_handle = None
        if clear_checkpoint:
            shutil.rmtree(self._ckpt_dir, ignore_errors=True)
        self._load_checkpoint()
        self.units = _UnitSink(self)
        (self.output/'export_config.json').write_text(json.dumps({
            'run_dirs': [str(p) for p in self.runs],'bootstrap_replicates': bootstrap_replicates,
            'permutation_replicates': permutation_replicates,'recompute_final': recompute_final,
            'recompute_layers': recompute_layers,'bootstrap_workers': self.bootstrap_workers},indent=2),encoding='utf-8')

    def _load_checkpoint(self):
        """Replay committed work units so an interrupted export can continue."""
        if not self._meta_path.is_file():
            return
        self._meta = json.loads(self._meta_path.read_text(encoding='utf-8'))
        committed = int(self._meta.get('seq', 0))
        for index in range(committed):
            path = self._units_dir/f'{index:05d}.pkl'
            if not path.is_file():
                raise RuntimeError(f'checkpoint unit {index} is missing: {path}')
            with path.open('rb') as handle:
                data = pickle.load(handle)
            for table, rows in data.get('tables', {}).items():
                self.tables[table].extend(rows)
            self.master.extend(data.get('master', []))
            self.artifacts.update(data.get('artifacts', {}))
            self._done.add(tuple(data['key']))
            del data
        self.next_result = int(self._meta.get('next_result', 0))
        self.log(f'Checkpoint: resumed {committed} committed unit(s), '
                 f'{len(self.master)} master row(s) replayed')
        gc.collect()

    def _write_meta(self):
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._meta_path.with_name('meta.json.tmp')
        tmp.write_text(json.dumps(self._meta), encoding='utf-8')
        os.replace(tmp, self._meta_path)

    def _write_unit_line(self, text):
        if self._unit_handle is not None:
            self._unit_handle.write(text)

    def unit_key(self, root, key):
        return (Path(root).name, *key)

    def unit_done(self, root, key):
        return self.unit_key(root, key) in self._done

    def begin_unit(self, root, key):
        """Open a checkpoint unit and snapshot the row counts it starts from."""
        self._unit = self.unit_key(root, key)
        self._unit_snapshot = {
            'tables': {table: len(rows) for table, rows in self.tables.items()},
            'master': len(self.master),
            'artifacts': set(self.artifacts),
        }
        self._units_dir.mkdir(parents=True, exist_ok=True)
        seq = int(self._meta.get('seq', 0))
        self._unit_handle = gzip.open(self._units_dir/f'{seq:05d}.units.jsonl.gz', 'wt', encoding='utf-8')  # noqa: SIM115
        self.log(f'  checkpoint unit open: {".".join(map(str, self._unit))} (#{seq})')

    def end_unit(self):
        """Persist the unit's added rows/artifacts, then commit ``meta.json``."""
        if self._unit is None:
            return
        if self._unit_handle is not None:
            self._unit_handle.close()
            self._unit_handle = None
        snapshot = self._unit_snapshot
        tables_diff = {}
        for table, rows in self.tables.items():
            start = snapshot['tables'].get(table, 0)
            if len(rows) > start:
                tables_diff[table] = rows[start:]
        master_diff = self.master[snapshot['master']:]
        artifacts_diff = {k: v for k, v in self.artifacts.items() if k not in snapshot['artifacts']}
        seq = int(self._meta.get('seq', 0))
        path = self._units_dir/f'{seq:05d}.pkl'
        tmp = path.with_name(f'{seq:05d}.pkl.tmp')
        with tmp.open('wb') as handle:
            pickle.dump({'key': list(self._unit), 'tables': tables_diff,
                         'master': master_diff, 'artifacts': artifacts_diff},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        self._meta['seq'] = seq + 1
        self._meta['next_result'] = self.next_result
        self._write_meta()
        added = sum(len(rows) for rows in tables_diff.values())
        self._done.add(self._unit)
        self.log(f'  checkpoint unit committed: {".".join(map(str, self._unit))} '
                 f'(+{added} row(s), {len(master_diff)} master)')
        self._unit = None
        self._unit_snapshot = None
        del tables_diff, master_diff, artifacts_diff, snapshot
        gc.collect()

    def abort_unit(self):
        """Discard the active (uncommitted) unit so an interrupted run leaks nothing."""
        if self._unit_handle is not None:
            self._unit_handle.close()
            self._unit_handle = None
        self._unit = None
        self._unit_snapshot = None

    def _assemble_source_units(self):
        """Rebuild ``source_units.jsonl.gz`` from committed per-unit streams."""
        path = self.output/'source_units.jsonl.gz'
        committed = int(self._meta.get('seq', 0))
        with gzip.open(path, 'wt', encoding='utf-8') as out:
            for index in range(committed):
                unit_path = self._units_dir/f'{index:05d}.units.jsonl.gz'
                if not unit_path.is_file():
                    continue
                with gzip.open(unit_path, 'rt', encoding='utf-8') as handle:
                    for line in handle:
                        out.write(line)

    def log(self, message):
        self._progress(f'[{time.time() - self._started:8.1f}s] {message}')

    def artifact(self, path):
        path = Path(path)
        key = str(path)
        if key not in self.artifacts:
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            self.artifacts[key] = {'artifact': key, 'sha256': h.hexdigest(), 'bytes': path.stat().st_size,
                                       'modified_ns': path.stat().st_mtime_ns}
        return key

    def missing(self, table, reason, **context):
        row = {k: None for k in FIELDS}
        row.update(context)
        row.update(status='missing', missing_reason=reason, estimate=None)
        self.tables[table].append(row)

    def emit(self, table, context, metric, value, low=None, high=None, **extra):
        row = {k: None for k in FIELDS}
        row.update(context)
        row.update(metric=metric, estimate=value, CI_low=low, CI_high=high)
        row.update(extra)
        row['status'] = row.get('status') or ('observed' if finite(value) else 'missing')
        row['statistics_status'] = row.get('statistics_status') or (
            'ci_available_p_not_computed' if finite(low) and finite(high) else 'ci_missing')
        identity = {k: row[k] for k in FIELDS if k not in ['result_id', 'estimate', 'CI_low', 'CI_high', 'FDR_q']}
        self.next_result += 1
        row['result_id'] = hashlib.sha256((json.dumps(identity, sort_keys=True, default=str)+f':{self.next_result}').encode()).hexdigest()[:24]
        self.master.append(row)
        self.tables[table].append(row)
        return row

    # Persisted stage rows carry an implicit representation space that must be
    # explicit in the export: a direction fitted on perturbation deltas and a
    # direction scored on raw ROI features are not interchangeable evidence.
    # ``source_direction_space`` is where the direction was fitted,
    # ``query_feature_space`` is where it was applied.
    _SPACES: ClassVar[dict] = {
        # E05 raw-ROI branch: M1 direction from delta space, scored on raw ROIs.
        ('E05', 'raw_roi'): ('unit_delta_M1', 'raw_ROI'),
        # E05 delta branch: fitted and scored in the same unit-delta space.
        ('E05', 'cross_perturbation_delta'): ('unit_delta_M1', 'unit_delta_M1'),
        # E05 comparators are fitted and scored entirely in raw ROI space.
        ('E05', 'raw_prototype'): ('raw_ROI', 'raw_ROI'),
        ('E05', 'clear_degraded_cav'): ('raw_ROI', 'raw_ROI'),
    }
    # E08 method -> (direction space, query/gallery space).  ``cfp_direction`` is
    # the M1/CMC residualized delta direction scored in its own delta-profile
    # space; ``cfp_direction_raw`` is the same profile in raw delta space;
    # ``cfp_direction_matched`` scores raw delta directions in raw ROI space.
    _E08_SPACES: ClassVar[dict] = {
        'cfp_direction': ('unit_delta_M1', 'delta_profile'),
        'cfp_direction_raw': ('unit_delta', 'delta_profile'),
        'cfp_direction_matched': ('unit_delta', 'raw_ROI'),
        'raw_prototype': ('raw_ROI', 'raw_ROI'),
        'cav_like': ('raw_ROI', 'raw_ROI'),
        'random': ('random', 'raw_ROI'),
        'text_baseline': ('text_shared_space', 'text_shared_space'),
    }

    @staticmethod
    def _roi_space(item):
        return 'raw_ROI_centered' if item.get('space') == 'centered' else 'raw_ROI'

    def _space_fields(self, stage, item):
        transfer = item.get('transfer')
        pair = self._SPACES.get((stage, transfer))
        if pair is None and stage == 'E04-1':
            pair = (self._roi_space(item), self._roi_space(item))
        if pair is None and stage == 'E03':
            pair = (self._roi_space(item), self._roi_space(item))
        if pair is None and stage == 'E07':
            # E07 readout features are either centered by the readout-fit global
            # mean (``space=centered``) or left uncentered as the M0 reference
            # (``space=raw``); the direction space depends on the control.
            # Delta-space CMC directions intervene on the readout features;
            # prototype/CAV/weight comparators are fitted in the same readout
            # feature space, so they follow the row's ``space``.
            condition = item.get('condition')
            if condition in ('prototype_derived', 'cav_derived',
                             'classifier_weight_aligned_random', 'linear_weight'):
                direction_space = self._roi_space(item)
            elif condition in ('wrong_raw', 'own_raw'):
                direction_space = 'unit_delta'
            elif condition == 'class_mean_axis':
                direction_space = 'unit_delta_M1'
            elif condition in ('random_unit', 'projection_matched_random'):
                direction_space = 'random'
            else:
                direction_space = 'unit_delta_M1'
            pair = (direction_space, self._roi_space(item))
        if pair is None:
            return {}
        source, query = pair
        return {'source_direction_space': source, 'query_feature_space': query,
                    'gallery_feature_space': None}

    def context(self, root, stage, item, path):
        config = self.config
        p = item.get('perturbation', item.get('source_perturbation'))
        strength = config.get('perturbations', {}).get(p, {})
        model = item.get('model_id')
        return dict(**self._space_fields(stage, item),
                    dataset=item.get('dataset_id'), model=model,
                    checkpoint=self.models.get(model, {}).get('checkpoint_sha256'),
                    layer=item.get('layer', 'final'), perturbation=p,
                    perturbation_strength=item.get('strength_value', strength),
                    operator=item.get('method', item.get('baseline', item.get('transfer'))),
                    normalization=item.get('normalization'), support_size=item.get('support_size', item.get('fit_size_limit')),
                    direction_fit_role='direction_fit', readout_fit_role='readout_fit', evaluation_role='confirmation',
                    split_seed=config['split']['seed'], run_id=root.name, stage=stage,
                    artifact=self.artifact(path), valid_n_objects=item.get('confirmation_object_count'),
                    n_source_images=item.get('confirmation_source_image_count', item.get('source_image_count')),
                    control=item.get('control'), analysis=item.get('analysis'),
                    source_perturbation=item.get('source_perturbation'), target_perturbation=item.get('target_perturbation'),
                    condition=item.get('condition'), alpha=item.get('alpha'), readout=item.get('readout'),
                    space=item.get('space'),
                    mean_abs_cos_h_w=item.get('mean_abs_cos_h_w'),
                    align_prototype=item.get('align_prototype'),
                    align_linear_weight=item.get('align_linear_weight'),
                    aggregation='persisted_stage_estimator; see source artifact',
                    status=item.get('status', 'observed'),
                    **({} if stage != 'E04-5' else {
                        'strength_parameter': item.get('strength_parameter'),
                        'strength_variant': item.get('strength_variant'),
                        'sweep_level': item.get('sweep_level'),
                        'variant_id': item.get('variant_id'),
                    }))

    def ingest(self, root, stage, item, path, table):
        context = self.context(root, stage, item, path)
        # Surface failed/missing source rows explicitly instead of silently
        # dropping them: they carry no metric keys, so the metric loop below
        # would otherwise emit nothing for them.
        if item.get('status') in ('missing', 'failed'):
            self.missing(table, item.get('missing_reason') or item.get('reason') or 'source row not completed', **context)
            return
        keys = list(METRICS)
        if stage == 'E07':
            # Keep the metric itself while excluding its own CI/count companion
            # fields; the interval is then recovered from ``<metric>_ci`` below,
            # so E07 no longer exports point estimates without uncertainty.
            keys += [k for k,v in item.items() if finite(v) and k.startswith(('frozen_', 'retrained_', 'prototype_priority'))
                     and not k.endswith(('_ci', '_ci_low', '_ci_high', '_count', '_bootstrap'))]
        for metric in dict.fromkeys(keys):
            if metric not in item:
                continue
            interval = item.get(metric+'_ci') or [item.get(metric+'_ci_low'), item.get(metric+'_ci_high')]
            if metric.endswith('_mean_l2'):
                interval = item.get(metric.replace('_mean_l2','_l2_ci')) or interval
            if metric == 'clean_degraded_delta_cosine_mean':
                interval = item.get('clean_degraded_delta_cosine_ci') or interval
            boot = (item.get('bootstrap') or {}).get(metric, {})
            if not isinstance(boot, dict): boot = {}
            row = self.emit(table, context, metric, item[metric], *interval,
                            bootstrap_replicates=boot.get('replicates'),
                            bootstrap_valid_replicates=boot.get('valid_replicates'),
                            p_value=item.get(metric+'_p_value'), FDR_q=item.get(metric+'_q_value'))
            row['evidence_version']='recomputed' if 'recomputed' in stage else 'persisted'
            if stage in ['E04','E04-4','E05'] and metric in ['own_argmax_rate','mean_diagonal_margin'] and not all(finite(v) for v in interval):
                row['missing_reason']='Persisted source-image CI is nonfinite; use recomputed evidence where available'
            if stage == 'E07':
                self.tables['readout_refit'].append(row)
                self.tables['removal_controls'].append(row)
            if stage == 'E04-4': self.tables['normalization'].append(row)
            if stage == 'specificity_recomputed': self.tables['normalization'].append(row)
        for metric, drop in (item.get('removal_drops') or {}).items():
            if isinstance(drop, dict):
                self.emit('removal_dose', context, metric, drop.get('estimate'), drop.get('ci_low'), drop.get('ci_high'))
        # Keep the complete persisted per-source payload, rather than flattening
        # away the class/source hierarchy or making up object-level scores.
        payload = item.get('source_metric_values')
        if payload:
            self.units.write(json.dumps({'context': context, 'source_metric_values': payload}, default=str)+'\n')

    def stages(self, root):
        from cfra.paper_export_contrasts import apply_contrasts
        mapping = {'E03':'baseline_comparison', 'E04':'specificity_full', 'E04-1':'baseline_comparison',
                   'E04-2':'support_scaling','E04-3':'context_factorial','E04-4':'specificity_full',
                   'E04-5':'strength_sweep','E05':'transfer_perturbation','E07':'removal_dose'}
        for stage, table in mapping.items():
            # Per-model E04-5 shards are recomputed sweep evidence and must not
            # enter the merged multi-seed export unless the caller opts in.
            if stage in self.exclude_stages:
                self.log(f'    {stage}: excluded by export configuration')
                continue
            paths = sorted((root/'intermediates/stage_results').glob(stage+'_*.json'))
            # Older runs wrote one whole-stage file; the current runner shards by
            # model.  Read whichever exists rather than assuming a single layout.
            legacy = root/'intermediates/stage_results'/f'{stage}.json'
            if legacy.is_file(): paths = [legacy, *paths]
            found = False
            stage_rows = 0
            for path in paths:
                if not path.exists(): continue
                data = read_json(path)
                if isinstance(data.get('models'), dict):
                    rows = [dict(row, model_id=m) for m, values in data['models'].items()
                            if isinstance(values, list) for row in values]
                else: rows = data.get('results', [])
                for item in rows:
                    target = 'raw_roi_transfer' if stage == 'E05' and item.get('transfer') != 'cross_perturbation_delta' else table
                    self.ingest(root, stage, item, path, target)
                found |= bool(rows)
                stage_rows += len(rows)
                # Persisted per-source-image statistics make paired contrasts
                # reconstructable without rerunning the stage: model/eval cell
                # differences are computed over the shared source-image units.
                apply_contrasts(self, stage, path, rows)
            if found:
                self.log(f'    {stage}: {stage_rows} source row(s) from {len(paths)} artifact(s)')
            else:
                self.missing(table, f'{stage} has no current stage artifact; pending/disabled', run_id=root.name, stage=stage)
                self.log(f'    {stage}: no current artifact (recorded as missing)')

    def retrieval(self, root):
        for path in sorted((root/'metrics').glob('E08_*.json')):
            # Cell artifacts and text summaries only; avoid double counting the
            # large model-level E08 aggregate containing those same cells.
            if path.stem in {f'E08_{m}' for m in self.models}: continue
            data = read_json(path)
            items = data.get('results',[]) if path.stem.startswith('E08_text_') else [data]
            self.log(f'    {path.name}: {len(items)} item(s)')
            for item in items:
                if not isinstance(item,dict) or not isinstance(item.get('methods'),dict): continue
                context = self.context(root,'E08',item,path)
                if path.stem.startswith('E08_text_'):
                    context.update(template_count=len(self.config.get('text_baseline_protocol',{}).get('prompt_templates',[])),
                                   prompt_templates=self.config.get('text_baseline_protocol',{}).get('prompt_templates',[]),
                                   class_name_mapping=self.config.get('text_baseline_protocol',{}).get('class_name_mapping'),
                                   analysis='ensemble')
                for method, scores in item['methods'].items():
                    direction_space, query_space = self._E08_SPACES.get(
                        method, (item.get('feature_key'), item.get('feature_key'))
                    )
                    ctx = dict(context, operator=method, query_count=item.get('query_count'), gallery_count=item.get('candidate_count'),
                               query_images=item.get('query_source_image_count'), gallery_images=item.get('candidate_source_image_count'),
                               image_disjoint=item.get('image_disjoint'),
                               query_feature_space=query_space, gallery_feature_space=query_space,
                               source_direction_space=direction_space,
                               space=scores.get('space'),
                               valid_n_objects=item.get('query_count'), n_source_images=item.get('query_source_image_count'))
                    for metric in ['macro_ap','precision_at_k','recall_at_k','ndcg_at_k']:
                        interval=scores.get(metric+'_ci') or [scores.get(metric+'_ci_low'),scores.get(metric+'_ci_high')]
                        self.emit('prompt_text_baseline' if method=='text_baseline' else 'retrieval',ctx,metric,scores.get(metric),*interval)
                    self.units.write(json.dumps({'context': ctx, 'rows': item.get('rows'), 'metrics': scores},default=str)+'\n')

    def population(self, root):
        images={}
        with (root/'manifest/images.csv').open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f): images[row['image_id']]=row
        population=defaultdict(Counter); classes=defaultdict(set); retained=defaultdict(set)
        role_counts=defaultdict(Counter); role_images=defaultdict(set)
        reasons=defaultdict(lambda: {'objects': 0,'images': set(),'classes': set()})
        class_counts=defaultdict(Counter)
        role_sets=defaultdict(lambda: defaultdict(set))
        reason_map={'minimum_object_side':'<10 px','object_side_exceeds_fixed_crop':'oversized object',
                    'image_pixels_exceed_limit':'>16,000,000 px image','geometry_control_infeasible':'geometry invalid',
                    'empty_visible_polygon':'empty polygon / visible polygon','difficulty_exceeds_limit':'unsupported difficulty'}
        for image in images.values():
            d=image['dataset']; population[d]['total_source_images']+=1
            population[d]['retained_images' if image['status']=='ok' else 'excluded_images']+=1
            if image['status']!='ok':
                reason=reason_map.get(image['status_reason'],'other')
                reasons[(d,reason)]['images'].add(image['image_id'])
            for field in ['image_id','content_group_id','image_sha256']:
                if image[field]:
                    role_sets[(d,image['role'])][field].add(image[field])
        with (root/'manifest/objects.csv').open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                d,c=row['dataset'],row['class_name']; classes[d].add(c); population[d]['total_objects']+=1
                keep=row['eligible'].lower()=='true'; key=(d,c)
                class_counts[key]['total_objects']+=1
                population[d]['retained_objects' if keep else 'excluded_objects']+=1
                class_counts[key]['retained_objects' if keep else 'excluded_objects']+=1
                if keep:
                    retained[d].add(c); r=row['role']; role_counts[(d,c,r)]['objects']+=1
                    role_images[(d,c,r)].add(row['image_id'])
                else:
                    reason=row['exclusion_reason'] or images[row['image_id']]['status_reason']
                    group=reason_map.get(reason,'other'); v=reasons[(d,group)]
                    v['objects']+=1; v['images'].add(row['image_id']); v['classes'].add(c)
        for d, counts in population.items():
            self.tables['population'].append(dict(dataset=d,record_type='population',**counts,
                total_classes=len(classes[d]),retained_classes=len(retained[d]),
                exclusion_percentage=100*counts['excluded_objects']/max(1,counts['total_objects']),
                percentage_denominator='total_objects',artifact=str(root/'manifest/objects.csv')))
            for reason in [*reason_map.values(),'other']:
                v=reasons[(d,reason)]
                self.tables['population'].append({'dataset': d,'record_type': 'exclusion_reason','reason': reason,
                    'removed_objects': v['objects'],'affected_images': len(v['images']),'affected_classes': len(v['classes']),
                    'percentage': 100*v['objects']/max(1,counts['total_objects']),'percentage_denominator': 'total_objects'})
        for (d,c),counts in class_counts.items():
            row=dict(dataset=d,class_name=c,**counts)
            for role in ROLES:
                row[role+'_objects']=role_counts[(d,c,role)]['objects']; row[role+'_images']=len(role_images[(d,c,role)])
            self.tables['split_roles'].append(row)
        for d in population:
            for a,b in itertools.combinations(ROLES,2):
                sets_a=role_sets[(d,a)]; sets_b=role_sets[(d,b)]
                self.tables['split_overlap'].append({'dataset': d,'role_a': a,'role_b': b,
                    'exact_source_image_overlap': len(sets_a['image_id']&sets_b['image_id']),
                    'content_group_overlap': len(sets_a['content_group_id']&sets_b['content_group_id']),
                    'duplicate_image_overlap': len(sets_a['image_sha256']&sets_b['image_sha256']),
                    'artifact': self.artifact(root/'manifest/images.csv')})
        self.artifact(root/'manifest/objects.csv'); self.artifact(root/'manifest/splits.csv')
        self.log(f'    population: {len(images)} images, '
                 f'{sum(population[d]["total_objects"] for d in population)} objects, '
                 f'datasets={sorted(population)}')

    def run(self):
        try:
            self.log(f'Export start: {len(self.runs)} run(s) -> {self.output}')
            for index, root in enumerate(self.runs, 1):
                self.log(f'Run {index}/{len(self.runs)}: {root}')
                handoff_path=root/'manifest/export_handoff.json'
                if handoff_path.exists():
                    self.log('  validating 50 export handoff')
                    handoff=read_json(handoff_path)
                    if handoff.get('status')!='ready_for_export_validation':
                        raise RuntimeError(f'50 handoff incomplete: {handoff_path}')
                    absent=[a['artifact'] for a in handoff['artifacts'] if not (root/a['artifact']).is_file()]
                    if absent:
                        raise RuntimeError(f'Artifacts invalidated after 50 completed: {absent}')
                self.config=read_json(root/'manifest/run.json')['resolved_config']
                self.active_run=root.name
                with (root/'manifest/models.csv').open(encoding='utf-8',newline='') as f:
                    self.models={row['model_id']:row for row in csv.DictReader(f)}
                interpolation=self.config.get('model_runtime',{}).get('positional_interpolation',{})
                if self.unit_done(root, ('registry',)):
                    self.log('  resume: model registry already checkpointed')
                else:
                    self.begin_unit(root, ('registry',))
                    self.tables['model_registry'].extend(dict(row,run_id=root.name) for row in self.models.values())
                    for row in self.tables['model_registry']:
                        m=row.get('model_id')
                        row.update(native_grid=interpolation.get('native_grid',{}).get(m),
                                   target_grid=interpolation.get('target_grid',{}).get(m),
                                   positional_interpolation=interpolation.get('method'))
                    self.artifact(root/'manifest/run.json'); self.artifact(root/'manifest/models.csv')
                    self.end_unit()
                self.log(f'  models={list(self.models)} datasets={self.config["scope"]["datasets"]} '
                         f'perturbations={self.config["scope"]["perturbations"]}')
                for key, label, method in (
                    (('population',), 'population manifest', 'population'),
                    (('stages',), 'persisted stage artifacts', 'stages'),
                    (('retrieval',), 'retrieval and text baselines', 'retrieval'),
                ):
                    if self.unit_done(root, key):
                        self.log(f'  resume: {label} already checkpointed'); continue
                    self.log(f'  {label}'); self.begin_unit(root, key); getattr(self, method)(root); self.end_unit()
                if self.recompute_final:
                    from cfra.paper_export_cache import export_final
                    self.log('  recompute final specificity'); export_final(self,root)
                if self.recompute_layers:
                    from cfra.paper_export_cache import export_layers
                    self.log('  recompute layerwise specificity'); export_layers(self,root)
                from cfra.paper_export_cache import export_layerwise_readout
                if self.unit_done(root, ('readout',)):
                    self.log('  resume: layerwise readout already checkpointed')
                else:
                    self.log('  layerwise readout intervention')
                    self.begin_unit(root, ('readout',)); export_layerwise_readout(self,root); self.end_unit()
                self.log(f'Run {index} done: master_rows={len(self.master)}')
            return self.finish()
        finally:
            self.abort_unit()
            self.units.close()

    # Measurement, provenance and run-scoped columns do not identify a
    # three-seed cell; everything else (including split_seed, which becomes the
    # per-seed estimate key) is part of the grouping.
    _NONIDENT = frozenset([
        'result_id', 'estimate', 'CI_low', 'CI_high', 'p_value', 'FDR_q',
        'valid_n_objects', 'n_source_images', 'bootstrap_replicates',
        'bootstrap_valid_replicates', 'permutation_replicates', 'statistics_status',
        'status', 'missing_reason', 'run_id', 'artifact', 'aggregation', 'family',
        'evidence_version', 'split_seed',
        # provenance / registry blobs are not identifying cell columns
        'path', 'sha256', 'bytes', 'modified_ns', 'asset_fingerprint', 'asset_id',
        'source_result_ids', 'loader_config_sha256', 'weight_path', 'record_type',
        'claim', 'claim_id', 'figure_table', 'source_table', 'section',
        'bootstrap', 'macro_ovr_auc_bootstrap', 'centered_prototype_ci',
        'centered_raw_ci', 'permutation_null',
    ])
    # Pure-provenance / summary tables are not seed-aggregated evidence.
    _SEED_SUMMARY_SKIP = frozenset([
        'artifact_registry', 'claim_registry', 'main_text_numbers',
        'statistical_summary', 'model_registry',
    ])

    @staticmethod
    def _hashable(value):
        """Coerce a cell value into a hashable grouping key component."""

        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        return value

    def seed_summary(self):
        """One row per identifying cell with the per-seed estimates and moments.

        Every exported table is aggregated across the supplied split seeds;
        missing estimates and non-identifying columns are excluded, never
        imputed.  The long per-run tables are left untouched.
        """
        columns_by_table = {}
        groups = defaultdict(lambda: defaultdict(list))
        for table in TABLES:
            if table in self._SEED_SUMMARY_SKIP:
                continue
            rows = self.tables[table]
            if not rows:
                continue
            extra = sorted({k for row in rows for k in row
                            if k not in FIELDS and k not in self._NONIDENT})
            columns = [c for c in FIELDS if c not in self._NONIDENT] + extra
            columns_by_table[table] = columns
            for row in rows:
                key = (table, *[self._hashable(row.get(c)) for c in columns])
                groups[key][self._hashable(row.get('split_seed'))].append(row.get('estimate'))
        summary = []
        for (table, *key), seed_map in groups.items():
            record = dict(zip(columns_by_table[table], key))
            record['table'] = table
            values = [v for estimates in seed_map.values() for v in estimates if finite(v)]
            record['n_seeds'] = len(seed_map)
            record['n_finite'] = len(values)
            record['mean'] = float(np.mean(values)) if values else None
            record['sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            record['min'] = float(np.min(values)) if values else None
            record['max'] = float(np.max(values)) if values else None
            for seed, estimates in seed_map.items():
                seed_values = [v for v in estimates if finite(v)]
                label = 'unset' if seed is None else seed
                record[f'seed_{label}'] = (float(np.mean(seed_values)) if seed_values
                                           else (estimates[0] if estimates else None))
            summary.append(record)
        summary.sort(key=lambda r: str(r.get('table')))
        return summary

    def finish(self):
        from cfra.paper_export_grid import expand_missing_grid
        if self._unit is not None:
            self.end_unit()
        self.log('Finish: assembling source_units from committed work units')
        self._assemble_source_units()
        self.log('Finish: expanding explicit missing seed/strength grid')
        expand_missing_grid(self)
        for table in TABLES:
            if not self.tables[table] and table not in ['statistical_summary','main_text_numbers','claim_registry','artifact_registry','seed_stability']:
                self.missing(table,'No compatible persisted result or reconstructable analysis exported; see coverage.json')
        # BH only within explicit, documented hypothesis families; missing p
        # values are excluded, never replaced with zero or inferred from a CI.
        families=defaultdict(list)
        for row in self.master:
            if finite(row.get('p_value')): families[row.get('family') or (row['stage'],row['metric'])].append(row)
        for rows in families.values():
            rows.sort(key=lambda r:r['p_value']); n=len(rows); q=1.
            for i in range(n-1,-1,-1):
                q=min(q,rows[i]['p_value']*n/(i+1)); rows[i]['FDR_q']=q
        self.log(f'Finish: BH across {len(families)} famil{"y" if len(families)==1 else "ies"}')
        from cfra.paper_export_views import build_views
        build_views(self)
        self.tables['statistical_summary']=self.master
        self.tables['artifact_registry']=list(self.artifacts.values())
        from cfra.paper_export_grid import REGISTERED_SEEDS
        seed_rows=[r for r in self.master if r['stage']=='E04-4' and r['control']=='true_obb']
        self.tables['seed_stability'].extend(seed_rows)
        for seed in REGISTERED_SEEDS:
            if not any(r['split_seed']==seed for r in seed_rows): self.missing('seed_stability','Seed has no supplied run',split_seed=seed)
        self.log(f'Finish: writing master + {len(TABLES)} numbered tables')
        write_csv(self.output/'00_master_results.csv',self.master,FIELDS)
        for i,name in enumerate(TABLES,1): write_csv(self.output/f'{i:02d}_{name}.csv',self.tables[name],FIELDS)
        write_csv(self.output/'tables_3seed'/'seed_summary.csv',self.seed_summary())
        write_csv(self.output/'main_text_numbers.csv',self.tables['main_text_numbers'])
        write_csv(self.output/'claim_registry.csv',self.tables['claim_registry'])
        coverage={name:{'rows': len(self.tables[name]),'missing': sum(r.get('status')=='missing' for r in self.tables[name])} for name in TABLES}
        coverage['statistical_completeness']={
            'master_rows':len(self.master),
            'finite_estimates':sum(finite(r['estimate']) for r in self.master),
            'finite_CI':sum(finite(r['CI_low']) and finite(r['CI_high']) for r in self.master),
            'finite_p':sum(finite(r['p_value']) for r in self.master),
            'finite_q':sum(finite(r['FDR_q']) for r in self.master),
        }
        (self.output/'coverage.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
        (self.output/'README.md').write_text(
            '# Paper export\n\n00_master_results.csv is the metric-level source of truth. '
            'Numbered tables and plot_data are views. Never average persisted and recomputed '
            'rows together: filter stage/evidence_version, operator, normalization, control, '
            'support, seed and feature space first.\n\n'
            'NaN means unavailable, not zero. coverage.json reports finite CI/p/q coverage. '
            'Missing experiments are explicitly identified. '
            'Prototype+CFP fusion and the legacy-style instance resplit are reconstructed from the '
            'run feature tables and are diagnostics, not reruns of the original protocol. '
            'No removal variant, missing layer or missing seed is fabricated.\n\n'
            'source_units.jsonl.gz preserves source-image summaries and reconstructed score matrices '
            '(rows=objects, columns=direction_classes). Reconstruct class/image aggregation from '
            'these scores, not from the persisted per-image own-win summaries: the estimators differ.\n\n'
            'New specificity p-values test class-to-direction correspondence under a shared random '
            'bijection of direction labels, one-sided. This is not an object-label permutation. '
            'BH families are explicitly named in family. Other tests stay missing until their '
            'paired units and null are specified. Marginal CIs are never subtracted to form a paired CI.\n\n'
            'Layerwise M6 is missing unless independent background layer pairs exist. '
            'Registered but unavailable layers (e.g. L1) must not be interpolated. '
            'Frozen/retrained E07 metrics retain their original names: recovery is not implicitly '
            'an AUC fraction. Claim links identify evidence, not an automatic validation of prose.\n',encoding='utf-8')
        self.log(f'Finish: exported {len(self.master)} master rows to {self.output}')
        from cfra.export_acceptance import write_acceptance
        acceptance=write_acceptance(self.output,self.tables,self.master)
        self.log(f'Finish: export acceptance={acceptance["status"]}')
        return acceptance
