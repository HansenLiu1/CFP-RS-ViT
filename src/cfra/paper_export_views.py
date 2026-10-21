"""Manuscript and plotting views reference result IDs, never hand-copied numbers."""
from collections import defaultdict

import numpy as np

from cfra.paper_export import FIELDS, finite, write_csv


def build_views(export):
    seed_groups=defaultdict(list)
    for row in export.master:
        if row['stage']=='specificity_recomputed' and finite(row['estimate']):
            key=tuple(row.get(k) for k in ['dataset','model','perturbation','operator','normalization','metric','layer'])
            seed_groups[key].append(row)
    for key,rows in seed_groups.items():
        values=np.asarray([r['estimate'] for r in rows])
        export.tables['seed_stability'].append(dict(zip(['dataset','model','perturbation','operator','normalization','metric','layer'],key),
            mean=float(values.mean()),SD=float(values.std(ddof=1)) if len(values)>1 else None,
            min=float(values.min()),max=float(values.max()),seed_range=float(np.ptp(values)),
            observed_seed_count=len({r['split_seed'] for r in rows}),status='single_seed' if len(rows)==1 else 'observed',
            source_result_ids=[r['result_id'] for r in rows]))
    claims=[
        ('C01','image-disjoint class association','specificity_full','Fig2'),
        ('C02','CMC improves specificity','specificity_full','Fig2/TableII'),
        ('C03','raw prototype standalone ranking','raw_roi_transfer','Fig3/TableII'),
        ('C04','cross-perturbation transfer','transfer_perturbation','Fig4'),
        ('C05','depth-dependent specificity','layerwise_specificity','Fig5'),
        ('C06','directional readout sensitivity','removal_dose','Fig6/TableIII'),
        ('C07','FAIR1M support boundary','support_scaling','Fig6'),
        ('C08','normalization dependence','normalization','FigS4'),
        ('C09','foreground-background interaction','context_factorial','context'),
        ('C10','strength stability','strength_sweep','FigS3'),
        ('C11','prototype+CFP fusion complementarity','prototype_cfp_complementarity','Fig3/TableII'),
        ('C12','legacy instance-level split leakage','legacy_split','TableS2'),
    ]
    for claim,text,table,figure in claims:
        for row in export.tables[table]:
            if not row.get('result_id'): continue
            record={'claim_id': claim,'claim': text,'figure_table': figure,'source_table': table,
                        'result_id': row['result_id'],'dataset': row['dataset'],'model': row['model'],
                        'metric': row['metric'],'estimate': row['estimate'],'CI_low': row['CI_low'],'CI_high': row['CI_high'],
                        'status': 'evidence_link_only_not_claim_validation'}
            export.tables['claim_registry'].append(record)
            export.tables['main_text_numbers'].append(dict(record,section=figure))
    plots={'Fig2':['specificity_full'],'Fig3':['baseline_comparison','raw_roi_transfer','retrieval'],
           'Fig4':['transfer_perturbation','raw_roi_transfer'], 'Fig5':['layerwise_specificity','layerwise_alignment'],
           'Fig6':['removal_dose','support_scaling'],'FigS1':['low_rank','common_mode_stability'],
           'FigS2':['specificity_classwise'],'FigS3':['strength_sweep'],'FigS4':['normalization'],
           'FigS5':['support_scaling'],'FigS6':['removal_dose','readout_refit'],
           'FigS7':['layerwise_specificity']}
    for figure,tables in plots.items():
        rows=[dict(row,source_table=table,metric_space=table) for table in tables for row in export.tables[table]]
        write_csv(export.output/'plot_data'/f'{figure}.csv',rows,['result_id','source_table','metric_space'])
    write_csv(export.output/'plot_data/Fig1_workflow.csv',[
        {'step': step,'status': 'illustration_required','artifact': None} for step in
        ['source_crop','OBB','clear_degraded','blur','gaussian_noise','patch_mask','brightness',
         'encoder','direction','CMC','specificity','transfer','removal']])
    # Paired changes of point estimates are descriptive until paired sampling
    # units have been aligned. Do not subtract marginal CI endpoints.
    for table,field,pairs in [('support_scaling','support_size',[(15,30),(30,50),(15,50)]),
                              ('normalization','normalization',[('unit','raw'),('unit','magnitude_weighted')])]:
        groups=defaultdict(dict)
        for row in list(export.tables[table]):
            if not row.get('result_id') or not finite(row.get('estimate')): continue
            key=tuple(row.get(k) for k in ['run_id','stage','dataset','model','perturbation','operator','control','analysis','metric','layer'])
            groups[key][row.get(field)]=row
        for group in groups.values():
            for a,b in pairs:
                if a not in group or b not in group: continue
                ra,rb=group[a],group[b]
                context={k:v for k,v in rb.items() if k not in ['result_id','metric','estimate','CI_low','CI_high']}
                context.update(analysis=f'delta_{a}_to_{b}',statistics_status='paired_units_required',
                               status='descriptive_difference',source_result_ids=[ra['result_id'],rb['result_id']])
                export.emit(table,context,'delta_'+rb['metric'],rb['estimate']-ra['estimate'])
    if not any(r.get('status')!='missing' for r in export.tables['prototype_cfp_complementarity']):
        export.missing('prototype_cfp_complementarity',
                       'Fusion requires a feature cache with readout-fit, validation and confirmation roles')
    if not any(r.get('status')!='missing' for r in export.tables['legacy_split']):
        export.missing('legacy_split',
                       'Legacy instance resplit requires a feature cache with object-level roles')
    export.missing('prompt_text_baseline','Only ensemble persisted; individual-template scores require separate evaluation',
                   analysis='individual_template')
    write_csv(export.output/'legacy_split_comparison.csv',export.tables['legacy_split'],FIELDS)
