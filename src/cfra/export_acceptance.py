"""Export completeness is stricter than successful CSV serialization."""
import json
import math


def write_acceptance(output, tables, master):
    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)
    gaps=[]
    required=['population','split_roles','split_overlap','legacy_split','specificity_full',
              'specificity_classwise','baseline_comparison','prototype_cfp_complementarity',
              'transfer_perturbation','raw_roi_transfer','retrieval',
              'layerwise_specificity','layerwise_alignment','strength_sweep','normalization',
              'support_scaling','context_factorial','seed_stability','removal_dose',
              'removal_controls','readout_refit','statistical_summary','main_text_numbers']
    for name in required:
        rows=tables[name]
        if not rows or all(r.get('status')=='missing' for r in rows):
            gaps.append(dict(table=name,reason='no_available_results'))
        for row in rows:
            if row.get('status')=='missing':
                gaps.append(dict(table=name,reason=row.get('missing_reason'),
                                 model=row.get('model'),stage=row.get('stage')))
    ci_metrics={'own_argmax_rate','mean_diagonal_margin','macro_ovr_auc','macro_ap',
                'precision_at_k','recall_at_k','ndcg_at_k','cross_perturbation_cosine'}
    missing_ci=sum(finite(r.get('estimate')) and r.get('metric') in ci_metrics and
                   not(finite(r.get('CI_low')) and finite(r.get('CI_high'))) for r in master)
    if missing_ci: gaps.append(dict(reason='nonfinite_CI_rows',count=missing_ci))
    missing_p=sum(finite(r.get('estimate')) and r.get('metric') in ci_metrics and
                  not finite(r.get('p_value')) for r in master)
    if missing_p: gaps.append(dict(reason='uncomputed_hypothesis_tests',count=missing_p))
    # Known implementation coverage must not disappear behind populated CSVs.
    observed_seeds={row.get('split_seed') for row in tables['seed_stability']
                    if row.get('status')!='missing' and isinstance(row.get('split_seed'),int)}
    if len(observed_seeds)<3:
        gaps.append(dict(reason='fewer than three registered split seeds supplied',
                         observed_seeds=sorted(observed_seeds)))
    if not any(row.get('status')!='missing' and finite(row.get('estimate'))
               for row in tables['strength_sweep']):
        gaps.append(dict(reason='E04-5 strength grid has no executed cells; strength robustness cannot be claimed'))
    if not any(row.get('status')!='missing' for row in tables['prompt_text_baseline']):
        gaps.append(dict(reason='text-baseline results are not available'))
    gaps.append(dict(reason='individual prompt-template scores are not evaluated separately from the ensemble'))
    gaps.append(dict(reason='curated manuscript aggregates and workflow image assets are incomplete'))
    result=dict(status='incomplete' if gaps else 'complete',csv_export_succeeded=True,
                export_ready=not gaps,gaps=gaps)
    (output/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
