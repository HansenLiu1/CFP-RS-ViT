"""Explicit missing seed/strength cells; observed results always take precedence."""
import json
from itertools import product

# Registered split seeds for the 3-seed paper plan.  Seeds outside this
# tuple are not required and must not be reported as missing coverage.
REGISTERED_SEEDS = (1729, 2718, 3141)


def _canonical_strength(value):
    """Normalize a strength value so int/float and dict-key forms compare equal.

    The main experiment records e.g. blur sigma as ``16.0`` while the declared
    grid may carry the integer ``16``; ``json.dumps`` would otherwise produce the
    distinct keys ``"16.0"`` and ``"16"`` and re-flag every observed cell.
    """

    if isinstance(value, dict):
        return {
            str(key): (float(item) if isinstance(item, (int, float)) and not isinstance(item, bool) else item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def _strength_key(value):
    return json.dumps(_canonical_strength(value), sort_keys=True)


def expand_missing_grid(export):
    metrics=['own_argmax_rate','mean_diagonal_margin','macro_ovr_auc']
    contexts=sorted({(r.get('dataset'),r.get('model'),r.get('perturbation')) for r in export.master
                     if r.get('dataset') and r.get('model') and r.get('perturbation')})
    seed_rows=[r for r in export.master if r.get('stage') in ['E04-4','specificity_recomputed']
               and r.get('control')=='true_obb' and r.get('operator')=='euclidean_mean_subtraction'
               and r.get('normalization')=='unit' and r.get('metric') in metrics]
    existing={(r['dataset'],r['model'],r['perturbation'],r['split_seed'],r['metric']) for r in seed_rows}
    for (dataset,model,pert),seed,metric in product(contexts,REGISTERED_SEEDS,metrics):
        if (dataset,model,pert,seed,metric) not in existing:
            export.emit('seed_stability',{'dataset':dataset,'model':model,'perturbation':pert,'split_seed':seed,
                'stage':'seed_grid','layer':'final','operator':'euclidean_mean_subtraction','normalization':'unit',
                'control':'true_obb','status':'missing','missing_reason':'No matching supplied seed result'},metric,None)
    grids={'blur':[('sigma_px',v) for v in [8.,16.,24.]],
           'gaussian_noise':[('sigma',v) for v in [.06,.12,.18]],
           'brightness':[('factor',v) for v in [.75,.55,.35]],
           'patch_mask':[('budget_side',{'target_fraction':a,'patch_side_fraction':b})
                         for a,b in product([.20,.35,.50],[.05,.10,.15])]}
    seeds=sorted({r['split_seed'] for r in export.master if isinstance(r.get('split_seed'),int) and r.get('run_id')})
    # Observed cells only count the manuscript headline cell (M1 unit true-OBB):
    # another operator satisfying the strength key must not mask a missing M1 row.
    observed={(r.get('dataset'),r.get('model'),r.get('perturbation'),r.get('split_seed'),
               _strength_key(r.get('perturbation_strength')),r.get('metric'))
              for r in export.tables['strength_sweep']
              if r.get('status')!='missing'
              and r.get('operator')=='euclidean_mean_subtraction'
              and r.get('normalization')=='unit'}
    for dataset,model,pert in contexts:
        for (parameter,value),seed,metric in product(grids.get(pert,[]),seeds,metrics):
            if (dataset,model,pert,seed,_strength_key(value),metric) in observed: continue
            export.emit('strength_sweep',{'dataset':dataset,'model':model,'perturbation':pert,'split_seed':seed,
                'perturbation_strength':value,'strength_parameter':parameter,'stage':'E04-5','layer':'final',
                'operator':'euclidean_mean_subtraction','normalization':'unit','status':'missing',
                'missing_reason':'No matching E04-5 strength cell; nominal main experiment is not a sweep result'},metric,None)
