"""Matched raw-ROI class-direction ranking, distinct from query/gallery retrieval."""
import json

import numpy as np

from cfra.methods.baselines import cav_direction, fit_clear_degraded_cav, fit_raw_prototype_directions

_LOG_DISCOUNT=np.log2(np.arange(2,12,dtype=np.float64))
_LOG_CUM=np.concatenate(([0.0],np.cumsum(1.0/_LOG_DISCOUNT)))


def _group_bounds(sorted_values,rows):
    """Tie-group start/end row per element for a 2-D column-sorted matrix."""
    start_flag=np.ones_like(sorted_values,dtype=bool); start_flag[1:]=sorted_values[1:]!=sorted_values[:-1]
    starts=np.where(start_flag,rows,-1); start_row=np.maximum.accumulate(starts,axis=0)
    end_flag=np.ones_like(sorted_values,dtype=bool); end_flag[:-1]=sorted_values[:-1]!=sorted_values[1:]
    ends=np.where(end_flag,rows,sorted_values.shape[0])
    end_row=np.minimum.accumulate(ends[::-1],axis=0)[::-1]
    return start_row,end_row


def _evaluate_matrix(matrix,relevance):
    """Macro over evaluable classes of AUC/AP/precision@10/recall@10/NDCG@10.

    All five metrics share one descending sort per column.  AUC uses the
    averaged-rank identity ``ascending_rank = (N+1) - descending_rank``, which
    is exact for tie groups, so no second sort is needed.  This replaces a
    per-class loop that called a sort, ``binary_auc`` and ``sklearn`` AP for
    every class on every one of the 10k+ permutation replicates.
    """
    n=matrix.shape[0]; rows=np.arange(n)[:,None]
    y=np.asarray(relevance,dtype=np.float64)
    total_pos=y.sum(axis=0); valid=(total_pos>0)&(total_pos<n)
    if not valid.any():
        return np.full(5,np.nan)
    order=np.argsort(-matrix,axis=0,kind="stable")
    sorted_s=np.take_along_axis(matrix,order,axis=0)
    sorted_y=np.take_along_axis(y,order,axis=0)
    start_row,end_row=_group_bounds(sorted_s,rows)
    start_flag=np.ones_like(sorted_s,dtype=bool); start_flag[1:]=sorted_s[1:]!=sorted_s[:-1]
    desc_rank=(start_row+end_row+2)/2.0
    n_pos=total_pos; n_neg=n-n_pos
    sum_desc_pos=(desc_rank*sorted_y).sum(axis=0)
    # AP: grouped precision/recall with tie groups.
    cum_pos=np.cumsum(sorted_y,axis=0)
    cum_end=np.take_along_axis(cum_pos,end_row,axis=0)
    cum_before=np.take_along_axis(cum_pos,np.maximum(start_row-1,0),axis=0)
    cum_before=np.where(start_row>0,cum_before,0.0)
    group_pos=cum_end-cum_before
    count_end=(end_row+1).astype(np.float64)
    k=min(10,n); topk=sorted_y[:k]; weights=1.0/_LOG_DISCOUNT[:k]
    precision=topk.mean(axis=0)
    dcg=(topk*weights[:,None]).sum(axis=0)
    ideal=_LOG_CUM[np.minimum(n_pos,k).astype(np.int64)]
    with np.errstate(invalid="ignore",divide="ignore"):
        auc=(n_pos*(n+1)-sum_desc_pos-n_pos*(n_pos+1)/2.0)/(n_pos*n_neg)
        ap=np.where(start_flag,(group_pos/n_pos)*(cum_end/count_end),0.0).sum(axis=0)
        recall=topk.sum(axis=0)/n_pos
        ndcg=np.where(ideal>0,dcg/ideal,np.nan)
    return np.stack([auc,ap,precision,recall,ndcg],axis=1)[valid].mean(axis=0)


def export_ranking(export,root,artifact,table,context,cfp,classes,ds,perts,roles,labels,images):
    select=(ds==context['dataset'])&(perts==context['perturbation'])
    fi=np.flatnonzero(select&(roles=='readout_fit')); qi=np.flatnonzero(select&(roles=='confirmation'))
    raw=np.asarray(table.arrays['raw'],dtype=float)
    proto=fit_raw_prototype_directions(raw[fi],labels[fi])
    supported=np.asarray([c for c in classes if c in proto and np.any(labels[fi]==c)])
    if len(supported)!=len(classes):
        export.missing('raw_roi_transfer','Baseline and CFP class supports differ; matched ranking not computed',**context)
        return
    prototypes=np.stack([proto[int(c)] for c in classes])
    cavs=np.stack([cav_direction(fit_clear_degraded_cav(
        table.arrays['true_clear_degraded'][fi[labels[fi]==c]],
        table.arrays['true_degraded_degraded'][fi[labels[fi]==c]],class_id=int(c))) for c in classes])
    norms=np.linalg.norm(raw[qi],axis=1); good=np.isfinite(raw[qi]).all(axis=1)&(norms>1e-8)
    qi=qi[good]; query=raw[qi]/norms[good,None]
    scores={name:query@directions.T for name,directions in
            [('CFP_CMC',cfp),('raw_prototype',prototypes),('clear_degraded_cav',cavs)]}
    # Binary relevance is fixed; a class without both positives and negatives in
    # a resample is dropped by _evaluate_matrix, exactly as the scalar loop did.
    relevance=(labels[qi][:,None]==classes[None,:]).astype(np.float64)
    names=['macro_ovr_auc','macro_ap','precision_at_10','recall_at_10','ndcg_at_10']
    unique=np.unique(images[qi]); groups=[np.flatnonzero(images[qi]==im) for im in unique]
    group_lengths=np.asarray([len(g) for g in groups],dtype=np.int64)
    group_offsets=np.concatenate(([0],np.cumsum(group_lengths)))
    flat_groups=np.concatenate(groups) if groups else np.empty(0,dtype=np.int64)
    stacked=np.stack([scores[name] for name in scores])
    rng=np.random.default_rng(context['split_seed'])
    # One draw per replicate on the caller RNG, matching the historical serial
    # sequence; the replicate loop then only resamples fixed row indices.
    draw_positions=rng.integers(0,len(groups),size=(export.bootstrap_replicates,len(groups)))
    draws={name:[] for name in scores}
    for row_positions in draw_positions:
        selected_lengths=group_lengths[row_positions]
        total=int(selected_lengths.sum())
        starts=group_offsets[row_positions]
        group_ids=np.repeat(np.arange(len(row_positions)),selected_lengths)
        within=np.arange(total)-np.repeat(np.cumsum(selected_lengths)-selected_lengths,selected_lengths)
        idx=flat_groups[starts[group_ids]+within]
        sampled=relevance[idx]
        sampled_matrices=stacked[:,idx]
        for k,name in enumerate(scores): draws[name].append(_evaluate_matrix(sampled_matrices[k],sampled))
    point={name:_evaluate_matrix(matrix,relevance) for name,matrix in scores.items()}
    base=dict(context,stage='raw_roi_ranking_recomputed',source_perturbation=context['perturbation'],
              query_feature_space='raw_ROI',gallery_feature_space='not_applicable_class_to_ROI_ranking',
              analysis='each class direction ranks confirmation ROIs; macro over classes',
              valid_n_objects=len(qi),n_source_images=len(unique))
    for name,score_matrix in scores.items():
        ctx=dict(base,operator=name,source_direction_space='unit_delta_M1' if name=='CFP_CMC' else 'raw_ROI',
                 readout_fit_role='readout_fit',evaluation_role='confirmation')
        boot=np.asarray(draws[name])
        for j,metric in enumerate(names):
            vals=boot[:,j];valid=vals[np.isfinite(vals)]
            lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
            export.emit('raw_roi_transfer',ctx,metric,float(point[name][j]),float(lo),float(hi),
                        bootstrap_replicates=export.bootstrap_replicates,bootstrap_valid_replicates=len(valid))
        export.units.write(json.dumps({"context": ctx,"source_image": images[qi].tolist(),
            "object_id": [table.metadata[i].object_id for i in qi],"class_id": labels[qi].tolist(),
            "direction_classes": classes.tolist(),"scores": score_matrix.tolist()},default=str)+'\n')
    # Shared image-cluster bootstrap draws preserve covariance for differences.
    for name in ['CFP_CMC','clear_degraded_cav']:
        differences=np.asarray(draws[name])-np.asarray(draws['raw_prototype'])
        exceed=np.zeros(5,dtype=int)
        observed=point[name]-point['raw_prototype']
        rng=np.random.default_rng(context['split_seed'])
        image_codes=np.searchsorted(unique,images[qi])
        higher=scores[name]; lower=scores['raw_prototype']
        left=np.empty_like(higher); right=np.empty_like(higher)
        for _ in range(export.permutation_replicates):
            swap=rng.integers(2,size=len(unique))[image_codes].astype(bool)
            np.copyto(left,lower); np.copyto(right,higher)
            left[swap]=higher[swap]; right[swap]=lower[swap]
            null=_evaluate_matrix(left,relevance)-_evaluate_matrix(right,relevance)
            exceed+=np.abs(null)>=np.abs(observed)-1e-12
        for j,metric in enumerate(names):
            valid=differences[:,j][np.isfinite(differences[:,j])]
            lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
            export.emit('baseline_comparison',dict(base,operator=name,analysis='paired_delta_vs_raw_prototype',
                family='raw_ROI_paired_vs_prototype:'+metric,permutation_null='swap method scores within source-image clusters; two-sided'),
                'delta_'+metric,float(observed[j]),float(lo),float(hi),
                p_value=float((exceed[j]+1)/(export.permutation_replicates+1)),
                permutation_replicates=export.permutation_replicates,bootstrap_valid_replicates=len(valid),
                bootstrap_replicates=export.bootstrap_replicates)
