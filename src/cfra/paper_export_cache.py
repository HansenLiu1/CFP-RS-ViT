"""Cache-only scientific analyses using the same operator fitting as the runner."""
from __future__ import annotations

import gc
import hashlib
import itertools
import json

import numpy as np

from cfra.evaluation.metrics import binary_auc
from cfra.methods.baselines import fit_raw_prototype_directions
from cfra.paper_export import finite, read_json
from cfra.pipeline.core import (
    _fit_operator_and_directions,
    _paired_auc_drop_ci,
    _safe_metrics,
    _score_metric,
    _score_metric_ci,
    _statistics_settings,
)
from cfra.pipeline.layerwise import load_layerwise_feature_artifact
from cfra.representations import cosine_scores

_ROLE_ORDER = ('direction_fit', 'readout_fit', 'validation', 'confirmation')
_FUSION_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
_METRIC_ALIASES = (('own_argmax_rate', 'own_argmax'), ('mean_diagonal_margin', 'margin'), ('macro_ovr_auc', 'auc'))


def unit(values, epsilon=1e-8):
    values=np.asarray(values,dtype=np.float64)
    norms=np.linalg.norm(values,axis=1)
    valid=np.isfinite(values).all(axis=1)&(norms>epsilon)
    return values[valid]/norms[valid,None],valid


def m1_origin(unit_values, labels, epsilon=1e-8):
    """M1/CMC origin: the mean of the per-class unit directions.

    This matches the residual operator the runner fits (``euclidean_mean_subtraction``
    over per-class unit directions), so descriptive analyses share the runner's
    centre instead of a per-object mean.
    """
    unit_values=np.asarray(unit_values,dtype=np.float64)
    labels=np.asarray(labels)
    directions=[]
    for c in np.unique(labels):
        block=unit_values[labels==c]
        if not len(block):
            continue
        mean=block.mean(axis=0)
        norm=np.linalg.norm(mean)
        if norm>epsilon:
            directions.append(mean/norm)
    if not directions:
        return None
    return np.mean(np.stack(directions),axis=0)


def bootstrap_mean_samples(units, replicates, seed, block=1024):
    """Resample cluster-level means with the exact serial RNG sequence.

    One ``rng.integers`` draw per replicate on the caller RNG reproduces the
    historical list-comprehension bootstrap bit-for-bit, while the block loop
    bounds peak memory for large replicate/cluster counts.
    """
    units=np.asarray(units,dtype=np.float64)
    replicates=int(replicates)
    if replicates<=0 or not len(units):
        return np.empty(0,dtype=np.float64)
    rng=np.random.default_rng(seed)
    parts=[]
    for start in range(0,replicates,block):
        size=min(block,replicates-start)
        positions=rng.integers(0,len(units),size=(size,len(units)))
        parts.append(units[positions].mean(axis=1))
    return np.concatenate(parts)


def export_layers(export, root):
    config=export.config
    settings=_statistics_settings(dict(config.get('statistics',{}),bootstrap_replicates=export.bootstrap_replicates,
                                       bootstrap_seed=config['split']['seed'],bootstrap_workers=export.bootstrap_workers))
    for model,dataset in itertools.product(config['scope']['models'],config['scope']['datasets']):
        key=('layers', model, dataset)
        if export.unit_done(root, key):
            export.log(f'Recompute layers: {model}/{dataset} (already checkpointed)'); continue
        artifact=root/'intermediates/features'/f'layerwise_{model}_{dataset}.npz'
        export.begin_unit(root, key)
        if not artifact.exists():
            export.missing('layerwise_specificity','Layerwise cache absent',model=model,dataset=dataset)
            export.end_unit()
            continue
        export.log(f'Recompute layers: {model}/{dataset}')
        table=load_layerwise_feature_artifact(root,artifact_name=artifact.stem)
        export.artifact(artifact)
        export.log(f'  layers loaded: {len(table.layers)}')
        for layer,record in table.layers.items():
            features=record['features']; metadata=record['metadata']
            delta=np.asarray(features['clear'],dtype=np.float64)-np.asarray(features['degraded'],dtype=np.float64)
            role=np.asarray([r['role'] for r in metadata]); labels=np.asarray([r['class_id'] for r in metadata])
            perts=np.asarray([r['perturbation'] for r in metadata]); images=np.asarray([r['source_image_id'] for r in metadata])
            export.log(f'  layer={layer} objects={len(metadata)}')
            alignment(export,root,artifact,table,model,dataset,layer,delta,role,labels,perts,images)
            for perturbation in config['scope']['perturbations']:
                fi=np.flatnonzero((role=='direction_fit')&(perts==perturbation))
                qi=np.flatnonzero((role=='confirmation')&(perts==perturbation))
                if not len(fi) or not len(qi): continue
                fit_unit,fv=unit(delta[fi]); query_unit,qv=unit(delta[qi])
                for normalization in config['protocol']['normalization_variants']:
                    if normalization=='unit': fit,query=fit_unit,query_unit
                    else:
                        scale=max(float(np.median(np.linalg.norm(delta[fi],axis=1))),1e-8) if normalization=='magnitude_weighted' else 1.
                        fit=delta[fi[fv]]/scale;query=delta[qi[qv]]/scale
                    for method in config['methods']['ids']:
                        if method=='background_svd_removal':
                            for metric in ['own_argmax_rate','mean_diagonal_margin','macro_ovr_auc']:
                                export.emit('layerwise_specificity',{'model': model,'dataset': dataset,'layer': layer,
                                    'perturbation': perturbation,'normalization': normalization,'operator': method,
                                    'split_seed': config['split']['seed'],'status': 'missing',
                                    'missing_reason': 'Independent layerwise background response cache absent'},metric,None)
                            continue
                        try:
                            op,dirs,classes=_fit_operator_and_directions(fit,labels[fi[fv]],method,background_fit_values=None,pca_rank=5,epsilon=1e-8)
                        except ValueError as exc:
                            export.missing('layerwise_specificity',str(exc),model=model,dataset=dataset,layer=layer,
                                           perturbation=perturbation,operator=method,normalization=normalization,
                                           missing_kind='analysis_failed')
                            continue
                        transformed=op.transform(query)
                        good=np.linalg.norm(transformed,axis=1)>1e-8
                        vals=transformed[good]; idx=qi[qv][good]
                        export.log(f'    {perturbation}/{normalization}/{method}: n={len(idx)}')
                        result=_safe_metrics(dirs,classes,vals,labels[idx],images[idx],statistics_config=settings,
                                             seed_parts=(root.name,model,dataset,layer,perturbation,normalization,method,'paper_export'))
                        item=dict(result,model_id=model,dataset_id=dataset,layer=layer,perturbation=perturbation,
                                  method=method,normalization=normalization,confirmation_object_count=len(idx))
                        export.ingest(root,'E06_recomputed',item,artifact,'layerwise_specificity')
                        scores=cosine_scores(vals,dirs)
                        context=export.context(root,'E06_recomputed',item,artifact)
                        export.units.write(json.dumps({'context': context,'source_image': images[idx].tolist(),
                            'class_id': labels[idx].tolist(),'direction_classes': classes.tolist(),'scores': scores.tolist()},default=str)+'\n')
                        classwise(export,context,scores,labels[idx],images[idx],classes)
                        mapping_permutation(export,context,scores,labels[idx],images[idx],classes)
            # BG-SVD must not silently substitute foreground response samples.
            export.missing('layerwise_specificity','M6 requires independent layerwise background pairs; paired cache stores foreground clear/degraded only',
                           model=model,dataset=dataset,layer=layer,operator='background_svd_removal')
        export.end_unit()
        del table
        gc.collect()


def _limit_class_budget(table, indices, limit):
    """Deterministically cap each class to ``limit`` objects by object-id hash."""
    if limit is None or limit <= 0:
        return np.asarray(indices, dtype=np.int64)
    by_class = {}
    for index in np.asarray(indices).tolist():
        by_class.setdefault(table.metadata[index].class_id, []).append(index)
    keep = []
    for class_indices in by_class.values():
        class_indices = sorted(
            class_indices,
            key=lambda index: (
                hashlib.sha256(table.metadata[index].object_id.encode("utf-8")).hexdigest(),
                table.metadata[index].object_id,
            ),
        )
        keep.extend(class_indices[:limit])
    return np.asarray(sorted(keep), dtype=np.int64)


def _apportion(count, fractions):
    """Largest-remainder allocation of ``count`` items over the role fractions."""
    total = float(sum(fractions)) or 1.0
    quotas = [count * value / total for value in fractions]
    counts = [int(np.floor(quota)) for quota in quotas]
    remainder = count - sum(counts)
    order = sorted(range(len(fractions)), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _legacy_role_plan(metadata, dataset, fractions, seed):
    """Object-level roles for the legacy-style instance resplit.

    Roles are assigned inside class and area-tertile strata without any
    source-image constraint, reproducing the same-image sharing that the
    image-disjoint protocol removes.  It is a diagnostic resplit of the common
    eligible pool that already carries features, not a rerun of the original
    protocol.
    """
    pools = {}
    for index, sample in enumerate(metadata):
        if sample.dataset_id != dataset:
            continue
        pools.setdefault(sample.class_id, {}).setdefault(sample.object_id, index)
    plan = {}
    for class_id, objects in sorted(pools.items()):
        items = sorted(objects.items())
        areas = np.asarray([metadata[index].geometry[0] for _, index in items], dtype=np.float64)
        order = np.argsort(areas, kind="stable")
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(f"legacy-split:{seed}:{dataset}:{class_id}".encode()).digest()[:4],
                "big",
            )
        )
        for stratum in np.array_split(order, 3):
            if len(stratum) == 0:
                continue
            shuffled = stratum[rng.permutation(len(stratum))]
            counts = _apportion(len(shuffled), fractions)
            start = 0
            for role, count in zip(_ROLE_ORDER, counts):
                for position in shuffled[start:start + int(count)]:
                    plan[items[int(position)][0]] = role
                start += int(count)
    return plan


def fusion(export, root, artifact, table, model, dataset, perturbation, cfp_directions, cfp_classes,
           labels, images, ds, perts, roles, settings):
    """Prototype-only vs CFP-only vs validation-selected fusion on raw ROI scores."""
    context = export.context(root, 'fusion_recomputed',
                             {'model_id': model, 'dataset_id': dataset, 'perturbation': perturbation}, artifact)
    context.update(query_feature_space='raw_ROI', source_direction_space='raw_ROI_or_unit_delta_M1',
                   analysis='validation_selected_zscore_fusion', aggregation='source-image clustered bootstrap')
    select = (ds == dataset) & (perts == perturbation)
    readout = np.flatnonzero(select & (roles == 'readout_fit'))
    validation = np.flatnonzero(select & (roles == 'validation'))
    confirmation = np.flatnonzero(select & (roles == 'confirmation'))
    if len(readout) == 0 or len(validation) == 0 or len(confirmation) == 0:
        export.missing('prototype_cfp_complementarity',
                       'Fusion requires readout-fit, validation and confirmation roles', **context)
        return
    raw = np.asarray(table.arrays['raw'], dtype=np.float64)
    try:
        prototypes = fit_raw_prototype_directions(raw[readout], labels[readout])
    except (ValueError, np.linalg.LinAlgError) as exc:  # BaselineError and near-zero directions
        export.missing('prototype_cfp_complementarity', f'Prototype fitting failed: {exc}', **context)
        return
    cfp = np.asarray(cfp_directions, dtype=np.float64)
    cfp_ids = [int(value) for value in np.asarray(cfp_classes).tolist()]
    shared = sorted(set(prototypes).intersection(cfp_ids))
    if len(shared) < 2:
        export.missing('prototype_cfp_complementarity',
                       'Fewer than two classes shared by prototype and CFP direction banks', **context)
        return
    classes = np.asarray(shared, dtype=np.int64)
    cfp_position = {class_id: index for index, class_id in enumerate(cfp_ids)}
    prototype_matrix = np.stack([prototypes[class_id] for class_id in shared])
    cfp_matrix = np.stack([cfp[cfp_position[class_id]] for class_id in shared])

    def scores(index):
        values, valid = unit(raw[index])
        return (cosine_scores(values, prototype_matrix), cosine_scores(values, cfp_matrix),
                np.asarray(index)[valid])

    p_validation, c_validation, index_validation = scores(validation)
    keep = np.isin(labels[index_validation], classes)
    p_validation, c_validation, index_validation = p_validation[keep], c_validation[keep], index_validation[keep]
    if len(index_validation) == 0:
        export.missing('prototype_cfp_complementarity',
                       'Validation role has no objects of the shared class set', **context)
        return
    mean_p = p_validation.mean(axis=0)
    std_p = np.where(p_validation.std(axis=0) > 1e-8, p_validation.std(axis=0), 1.0)
    mean_c = c_validation.mean(axis=0)
    std_c = np.where(c_validation.std(axis=0) > 1e-8, c_validation.std(axis=0), 1.0)
    validation_labels = labels[index_validation]
    validation_aucs = []
    for weight in _FUSION_LAMBDAS:
        fused = (1.0 - weight) * (p_validation - mean_p) / std_p + weight * (c_validation - mean_c) / std_c
        validation_aucs.append(_score_metric(fused, validation_labels, classes, 'auc'))
    weight = float(_FUSION_LAMBDAS[int(np.nanargmax(validation_aucs))]) if np.isfinite(validation_aucs).any() else 0.0

    p_confirmation, c_confirmation, index_confirmation = scores(confirmation)
    keep = np.isin(labels[index_confirmation], classes)
    p_confirmation, c_confirmation, index_confirmation = (
        p_confirmation[keep], c_confirmation[keep], index_confirmation[keep])
    if len(index_confirmation) == 0:
        export.missing('prototype_cfp_complementarity',
                       'Confirmation role has no objects of the shared class set', **context)
        return
    z_prototype = (p_confirmation - mean_p) / std_p
    z_cfp = (c_confirmation - mean_c) / std_c
    z_fusion = (1.0 - weight) * z_prototype + weight * z_cfp
    confirmation_labels = labels[index_confirmation]
    confirmation_images = images[index_confirmation]
    context.update(fusion_weight=weight, fusion_weight_grid=json.dumps(list(_FUSION_LAMBDAS)),
                   selection_role='validation', selection_metric='macro_ovr_auc',
                   validation_objects=len(index_validation),
                   validation_source_images=len(set(images[index_validation].tolist())),
                   confirmation_objects=len(index_confirmation),
                   confirmation_source_images=len(set(confirmation_images.tolist())),
                   valid_n_objects=len(index_confirmation),
                   n_source_images=len(set(confirmation_images.tolist())))
    for name, scores_matrix in (('raw_prototype', z_prototype), ('cfp_cmc', z_cfp),
                                ('prototype_cfp_fusion', z_fusion)):
        method_context = dict(
            context, operator=name,
            analysis='validation_selected_zscore_fusion' if name == 'prototype_cfp_fusion' else 'standalone_matched_raw_space')
        for output_metric, core_metric in _METRIC_ALIASES:
            point = _score_metric(scores_matrix, confirmation_labels, classes, core_metric)
            ci = _score_metric_ci(scores_matrix, confirmation_labels, confirmation_images, classes, core_metric,
                                  settings=settings,
                                  seed_parts=(root.name, model, dataset, perturbation, 'fusion', name, output_metric))
            export.emit('prototype_cfp_complementarity', method_context, output_metric, point,
                        ci.get('ci_low'), ci.get('ci_high'),
                        bootstrap_replicates=ci.get('replicates'), bootstrap_valid_replicates=ci.get('replicates'))
    for name, current in (('fusion_minus_prototype', z_prototype), ('fusion_minus_cfp', z_cfp)):
        gain = _paired_auc_drop_ci(z_fusion, current, confirmation_labels, confirmation_images, classes,
                                   settings=settings,
                                   seed_parts=(root.name, model, dataset, perturbation, 'fusion_gain', name))
        export.emit('prototype_cfp_complementarity',
                    dict(context, operator='prototype_cfp_fusion', analysis=name),
                    name + '_macro_ovr_auc', gain.get('estimate'), gain.get('ci_low'), gain.get('ci_high'),
                    bootstrap_replicates=gain.get('replicates'))


def legacy_resplit(export, root, artifact, table, model, dataset, perturbation, plan, canonical, settings,
                   primary_fit_size):
    """Legacy-style instance-level resplit over the common eligible object pool."""
    context = export.context(root, 'legacy_split_recomputed',
                             {'model_id': model, 'dataset_id': dataset, 'perturbation': perturbation}, artifact)
    context.update(protocol='legacy_instance_resplit',
                   analysis='object_level_resplit_of_common_eligible_pool',
                   aggregation='legacy instance roles; source images may repeat across roles')
    delta = (np.asarray(table.arrays['true_clear_degraded'], dtype=np.float64)
             - np.asarray(table.arrays['true_degraded_degraded'], dtype=np.float64))
    indices = np.asarray([index for index, sample in enumerate(table.metadata)
                          if sample.dataset_id == dataset and sample.perturbation == perturbation], dtype=np.int64)
    roles = np.asarray([plan.get(table.metadata[index].object_id, '') for index in indices.tolist()], dtype='U')
    fit_index = _limit_class_budget(table, indices[roles == 'direction_fit'], primary_fit_size)
    query_index = indices[roles == 'confirmation']
    if len(fit_index) == 0 or len(query_index) == 0:
        export.missing('legacy_split', 'Legacy resplit produced no direction-fit or confirmation objects', **context)
        return
    fit, fit_valid = unit(delta[fit_index])
    fit_index = fit_index[fit_valid]
    fit_labels = np.asarray([table.metadata[index].class_id for index in fit_index.tolist()], dtype=np.int64)
    query, query_valid = unit(delta[query_index])
    query_index = query_index[query_valid]
    try:
        operator, directions, classes = _fit_operator_and_directions(
            fit, fit_labels, 'euclidean_mean_subtraction',
            background_fit_values=None, pca_rank=5, epsilon=1e-8)
    except ValueError as exc:
        export.missing('legacy_split', str(exc), **context)
        return
    values = operator.transform(query)
    keep = np.linalg.norm(values, axis=1) > 1e-8
    values, query_index = values[keep], query_index[keep]
    confirmation_labels = np.asarray([table.metadata[index].class_id for index in query_index.tolist()], dtype=np.int64)
    confirmation_images = np.asarray([table.metadata[index].source_image_id for index in query_index.tolist()], dtype='U')
    metrics = _safe_metrics(directions, classes, values, confirmation_labels, confirmation_images,
                            statistics_config=settings,
                            seed_parts=(root.name, model, dataset, perturbation, 'legacy_instance_resplit'))
    fit_images = {table.metadata[index].source_image_id for index in fit_index.tolist()}
    query_images = set(confirmation_images.tolist())
    shared = len(fit_images & query_images)
    context.update(fit_objects=len(fit_index), fit_source_images=len(fit_images),
                   confirmation_objects=len(query_index), confirmation_source_images=len(query_images),
                   shared_source_images=shared,
                   shared_source_image_fraction=float(shared / max(1, len(query_images))),
                   primary_fit_size=int(primary_fit_size),
                   direction_fit_role='legacy_direction_fit', readout_fit_role='legacy_readout_fit',
                   evaluation_role='legacy_confirmation', query_feature_space='unit_delta_M1')
    for metric in ('own_argmax_rate', 'mean_diagonal_margin', 'macro_ovr_auc'):
        export.emit('legacy_split', dict(context, analysis='legacy_instance_resplit'), metric,
                    metrics.get(metric), metrics.get(metric + '_ci_low'), metrics.get(metric + '_ci_high'))
        if canonical and finite(canonical.get(metric)):
            export.emit('legacy_split',
                        dict(context, analysis='canonical_minus_legacy',
                             statistics_status='descriptive_difference'),
                        'canonical_minus_legacy_' + metric, float(canonical[metric] - metrics.get(metric)))
    outcomes = {name: [] for name in ('same_image', 'cross_image')}
    for start in range(0, len(values), 128):
        end = min(start + 128, len(values))
        matrix = values[start:end] @ values.T
        same = confirmation_images[start:end, None] == confirmation_images[None, :]
        self_pair = np.arange(start, end)[:, None] == np.arange(len(values))[None, :]
        masks = {'same_image': same & ~self_pair, 'cross_image': ~same}
        for name, mask in masks.items():
            count = mask.sum(axis=1)
            outcomes[name].extend(
                np.divide((matrix * mask).sum(axis=1), count,
                          out=np.full(end - start, np.nan), where=count > 0))
    for name, result in outcomes.items():
        image_mean_ci(export, dict(context, analysis=name), 'pair_cosine',
                      np.asarray(result), confirmation_images, 'legacy_split')


def export_final(export, root):
    from cfra.pipeline.core import _limit_fit_mask
    from cfra.pipeline.features import load_feature_table
    config=export.config
    settings=_statistics_settings(dict(config.get('statistics',{}),bootstrap_replicates=export.bootstrap_replicates,
                                       bootstrap_seed=config['split']['seed'],bootstrap_workers=export.bootstrap_workers))
    for model in config['scope']['models']:
        key=('final', model)
        if export.unit_done(root, key):
            export.log(f'Recompute final specificity: {model} (already checkpointed)'); continue
        export.log(f'Recompute final specificity: {model}')
        export.begin_unit(root, key)
        table=load_feature_table(root,model)
        export.log(f'  loaded feature table: {len(table.metadata)} objects')
        for feature_path in (root/'intermediates/features').glob(f'*{model}*'):
            if feature_path.is_file() and not feature_path.name.startswith('layerwise_'):
                export.artifact(feature_path)
        artifact=root/'manifest'/f'features_{model}.json'
        from cfra.pipeline.core import run_e04
        export.log(f'  spatial controls (true_obb skipped): {model}')
        control_rows=run_e04(table,dataset_ids=config['scope']['datasets'],
            perturbations=config['scope']['perturbations'],method_ids=config['methods']['ids'],
            normalization_variants=config['protocol']['normalization_variants'],epsilon=1e-8,
            fit_size=config['protocol']['primary_fit_size'],controls=config.get('controls'),
            statistics_config=settings,random_seed=config['split']['seed'],include_true_obb=False,
            progress=export.log)
        for row in control_rows:
            if row.get('control')!='true_obb':
                export.ingest(root,'spatial_control_recomputed',row,artifact,'specificity_full')
        del control_rows
        labels=np.asarray([r.class_id for r in table.metadata]); images=np.asarray([r.source_image_id for r in table.metadata])
        ds=np.asarray([r.dataset_id for r in table.metadata]); perts=np.asarray([r.perturbation for r in table.metadata]); roles=np.asarray([r.role for r in table.metadata])
        delta=table.arrays['true_clear_degraded']-table.arrays['true_degraded_degraded']
        bg=table.arrays['background_clear']-table.arrays['background_degraded']
        legacy_plans={dataset:_legacy_role_plan(table.metadata,dataset,config['split']['fractions'],config['split']['seed'])
                      for dataset in config['scope']['datasets']}
        for dataset,pert in itertools.product(config['scope']['datasets'],config['scope']['perturbations']):
            select=(ds==dataset)&(perts==pert)
            fit_mask=_limit_fit_mask(table,select&(roles=='direction_fit'),config['protocol']['primary_fit_size'])
            fi=np.flatnonzero(fit_mask); qi=np.flatnonzero(select&(roles=='confirmation'))
            export.log(f'  {model}/{dataset}/{pert}: fit={len(fi)} confirm={len(qi)}')
            fit_unit,fv=unit(delta[fi])
            origin=m1_origin(fit_unit,labels[fi][fv])
            lowrank(export,root,artifact,model,dataset,pert,delta[fi],origin=origin)
            context_factorial(export,root,artifact,table,model,dataset,pert,qi,images)
            stability(export,root,artifact,model,dataset,pert,delta[qi],labels[qi],images[qi],origin=origin)
            unit_cfp_dirs=None; unit_cfp_classes=None; unit_canonical=None
            for normalization in config['protocol']['normalization_variants']:
                # Magnitude weighting is raw / median fit norm, not a per-query scale.
                scale=max(float(np.median(np.linalg.norm(delta[fi],axis=1))),1e-8)
                if normalization=='unit': fit,fv=unit(delta[fi]); query,qv=unit(delta[qi]); background,_=unit(bg[fi])
                else:
                    divisor=scale if normalization=='magnitude_weighted' else 1.
                    fit=np.asarray(delta[fi],dtype=float)/divisor; query=np.asarray(delta[qi],dtype=float)/divisor
                    fv=np.linalg.norm(fit,axis=1)>1e-8; qv=np.linalg.norm(query,axis=1)>1e-8
                    fit=fit[fv];query=query[qv];background=np.asarray(bg[fi],dtype=float)/divisor
                for method in config['methods']['ids']:
                    export.log(f'    {normalization}/{method}')
                    try:
                        op,dirs,classes=_fit_operator_and_directions(fit,labels[fi[fv]],method,background_fit_values=background,pca_rank=5,epsilon=1e-8)
                    except ValueError as exc:
                        export.missing('specificity_full',str(exc),model=model,dataset=dataset,perturbation=pert,
                                       operator=method,normalization=normalization,stage='specificity_recomputed')
                        continue
                    vals=op.transform(query); good=np.linalg.norm(vals,axis=1)>1e-8; vals=vals[good]; idx=qi[qv][good]
                    result=_safe_metrics(dirs,classes,vals,labels[idx],images[idx],statistics_config=settings,
                        seed_parts=(root.name,model,dataset,pert,normalization,method,'export_final'))
                    item=dict(result,model_id=model,dataset_id=dataset,perturbation=pert,method=method,
                        normalization=normalization,control='true_obb',fit_size_limit=config['protocol']['primary_fit_size'],confirmation_object_count=len(idx))
                    export.ingest(root,'specificity_recomputed',item,artifact,'specificity_full')
                    context=export.context(root,'specificity_recomputed',item,artifact); scores=cosine_scores(vals,dirs)
                    export.units.write(json.dumps({'context': context,'source_image': images[idx].tolist(),'class_id': labels[idx].tolist(),
                        'object_id': [table.metadata[i].object_id for i in idx],'direction_classes': classes.tolist(),'scores': scores.tolist()},default=str)+'\n')
                    if normalization=='unit' and method=='euclidean_mean_subtraction':
                        from cfra.paper_export_ranking import export_ranking
                        export_ranking(export,root,artifact,table,context,dirs,classes,ds,perts,roles,labels,images)
                        transfer(export,root,artifact,table,context,op,dirs,classes,delta,ds,perts,roles,labels,images,settings)
                        unit_cfp_dirs=dirs; unit_cfp_classes=classes; unit_canonical=result
                    classwise(export,context,scores,labels[idx],images[idx],classes)
                    mapping_permutation(export,context,scores,labels[idx],images[idx],classes)
            if unit_canonical is not None:
                fusion(export,root,artifact,table,model,dataset,pert,unit_cfp_dirs,unit_cfp_classes,
                       labels,images,ds,perts,roles,settings)
            legacy_plan=legacy_plans.get(dataset)
            if legacy_plan:
                legacy_resplit(export,root,artifact,table,model,dataset,pert,legacy_plan,unit_canonical,
                               settings,config['protocol']['primary_fit_size'])
        export.end_unit()
        del table, labels, images, ds, perts, roles, delta, bg, legacy_plans
        gc.collect()


def lowrank(export,root,artifact,model,dataset,pert,values,origin=None):
    def ranks(x):
        eigen=np.linalg.svd(x,compute_uv=False)**2
        weights=eigen/eigen.sum(); weights=weights[weights>0]
        return float(1/(weights@weights)),float(np.exp(-(weights*np.log(weights)).sum()))
    rng=np.random.default_rng(export.config['split']['seed'])
    raw,_=unit(values)
    reference,_=unit(rng.standard_normal(raw.shape))
    rpr,rer=ranks(reference)
    if origin is None:
        origin=raw.mean(axis=0) if len(raw) else np.zeros(raw.shape[1])
    centered,_=unit(raw-origin)
    base=export.context(root,'low_rank_recomputed',{'model_id': model,'dataset_id': dataset,'perturbation': pert},artifact)
    variants=[('raw',raw,'uncentered response second moment; matched isotropic unit Gaussian reference'),
              ('cmc',centered,'M1/CMC-origin centered response second moment; matched isotropic unit Gaussian reference')]
    for space,matrix,analysis in variants:
        if not len(matrix):
            export.missing('low_rank','No finite direction_fit unit deltas after origin shift',
                           **dict(base,space=space))
            continue
        pr,er=ranks(matrix)
        context=dict(base,space=space,evaluation_role='direction_fit',normalization='unit',
                     valid_n_objects=len(matrix),analysis=analysis)
        for metric,value in [('response_dimension',matrix.shape[1]),('participation_ratio',pr),('effective_rank',er),
                             ('random_participation_ratio',rpr),('random_effective_rank',rer)]:
            export.emit('low_rank',context,metric,value)


def transfer(export,root,artifact,table,context,op,dirs,classes,delta,ds,perts,roles,labels,images,settings):
    for target in export.config['scope']['perturbations']:
        idx=np.flatnonzero((ds==context['dataset'])&(perts==target)&(roles=='confirmation'))
        values,valid=unit(delta[idx]); idx=idx[valid]; values=op.transform(values)
        good=np.linalg.norm(values,axis=1)>1e-8; idx=idx[good];values=values[good]
        metrics=_safe_metrics(dirs,classes,values,labels[idx],images[idx],statistics_config=settings,
            seed_parts=(context['run_id'],context['model'],context['dataset'],context['perturbation'],target,'transfer_export'))
        item=dict(metrics,model_id=context['model'],dataset_id=context['dataset'],method=context['operator'],normalization='unit',
                  source_perturbation=context['perturbation'],target_perturbation=target,confirmation_object_count=len(idx),
                  analysis='same_perturbation' if target==context['perturbation'] else 'cross_perturbation')
        export.ingest(root,'transfer_recomputed',item,artifact,'transfer_perturbation')
        scores=cosine_scores(values,dirs)
        export.units.write(json.dumps({'context': export.context(root,'transfer_recomputed',item,artifact),
            'source_image': images[idx].tolist(),'class_id': labels[idx].tolist(),'direction_classes': classes.tolist(),'scores': scores.tolist()},default=str)+'\n')


def image_mean_ci(export,context,metric,values,images,table):
    valid=np.isfinite(values)
    values=np.asarray(values)[valid]; images=np.asarray(images)[valid]
    units=np.asarray([values[images==im].mean() for im in np.unique(images)])
    if not len(units):
        export.missing(table,'No finite source-image units',**context); return
    draws=bootstrap_mean_samples(units,export.bootstrap_replicates,context['split_seed'])
    lo,hi=np.quantile(draws,[.025,.975]) if len(draws) else (np.nan,np.nan)
    export.emit(table,dict(context,n_source_images=len(units),valid_n_objects=len(values),
                aggregation='object -> source image -> mean',bootstrap_replicates=len(draws),bootstrap_valid_replicates=len(draws)),
                metric,float(units.mean()),float(lo),float(hi))
    export.units.write(json.dumps({'context': context,'metric': metric,'source_image': images.tolist(),'raw_value': values.tolist()},default=str)+'\n')


def context_factorial(export,root,artifact,table,model,dataset,pert,idx,images):
    from cfra.pipeline.core import _class_mean_pair_cosine
    arrays=table.arrays
    keys={'FcBc':'true_clear_clean','FdBc':'true_degraded_clean','FcBd':'true_clear_degraded','FdBd':'true_degraded_degraded'}
    context=export.context(root,'factorial_recomputed',{'model_id': model,'dataset_id': dataset,'perturbation': pert},artifact)
    if any(key not in arrays for key in keys.values()):
        export.missing('context_factorial','Missing four-state feature arrays',**context); return
    meta=table.metadata
    roles=np.asarray([r.role for r in meta]); dss=np.asarray([r.dataset_id for r in meta]); per=np.asarray([r.perturbation for r in meta])
    labels=np.asarray([r.class_id for r in meta])
    fit=np.flatnonzero((dss==dataset)&(per==pert)&(roles=='direction_fit'))
    raw_states={name:np.asarray(arrays[key][idx],dtype=float) for name,key in keys.items()}
    # Centered variant: remove each state's own direction-fit mean (its per-state
    # common mode) so the two backgrounds are compared after that shared origin
    # is subtracted.  This per-state origin is deliberately NOT the M1 origin
    # used by low_rank/common_mode_stability (the mean of the per-class unit
    # directions); it is labelled ``centered`` to match the persisted E04-3 rows
    # and must not be pooled with the ``cmc`` space of those other tables.  The
    # ``raw`` variant is the M0 reference.
    cmc_states={name:np.asarray(arrays[key][idx],dtype=float)
                     - (np.asarray(arrays[key][fit],dtype=float).mean(axis=0) if len(fit) else 0.0)
                for name,key in keys.items()}
    for space,states in (('raw',raw_states),('centered',cmc_states)):
        clean=states['FcBc']-states['FdBc']; degraded=states['FcBd']-states['FdBd']
        contrasts={'d_Bc':clean,'d_Bd':degraded,'interaction':clean-degraded}
        for name,values in {**states,**contrasts}.items():
            image_mean_ci(export,dict(context,space=space,analysis=name),'mean_feature_norm',np.linalg.norm(values,axis=1),images[idx],'context_factorial')
        denominator=np.linalg.norm(clean,axis=1)*np.linalg.norm(degraded,axis=1)
        cosine=np.divide((clean*degraded).sum(axis=1),denominator,out=np.full(len(idx),np.nan),where=denominator>1e-16)
        image_mean_ci(export,dict(context,space=space,analysis='d_Bc_vs_d_Bd'),'direction_cosine',cosine,images[idx],'context_factorial')
        # Class-level counterpart: mean over classes of the cosine between the
        # per-class mean d_Bc and d_Bd.  Object-level cosines are noise-dominated.
        class_mean_cosine=_class_mean_pair_cosine(clean,degraded,labels[idx])
        export.emit('context_factorial',dict(context,space=space,analysis='d_Bc_vs_d_Bd_class_mean'),
                    'direction_cosine_class_mean',float(class_mean_cosine))


def stability(export,root,artifact,model,dataset,pert,values,labels,images,origin=None):
    context=export.context(root,'stability_recomputed',{'model_id': model,'dataset_id': dataset,'perturbation': pert},artifact)

    def outcomes_for(matrix,matrix_labels,matrix_images):
        # Each query image is a cluster. A bounded matrix block avoids an N^2
        # allocation; exact sums retain every eligible ordered pair.
        outcomes={k:[] for k in ['same_image','cross_image','same_class_cross_image','different_class_cross_image']}
        for start in range(0,len(matrix),128):
            end=min(start+128,len(matrix)); scores=matrix[start:end]@matrix.T
            same_image=matrix_images[start:end,None]==matrix_images[None,:]
            same_class=matrix_labels[start:end,None]==matrix_labels[None,:]
            self_pair=np.arange(start,end)[:,None]==np.arange(len(matrix))[None,:]
            masks={'same_image':same_image&~self_pair,'cross_image':~same_image,
                   'same_class_cross_image':~same_image&same_class,'different_class_cross_image':~same_image&~same_class}
            for name,mask in masks.items():
                count=mask.sum(axis=1)
                outcomes[name].extend(np.divide((scores*mask).sum(axis=1),count,out=np.full(end-start,np.nan),where=count>0))
        return outcomes

    raw,raw_valid=unit(values)
    # ``unit`` returns a compressed matrix plus a mask over its input, so the
    # label/image arrays must be advanced through the same masks before reuse.
    raw_labels=labels[raw_valid]; raw_images=images[raw_valid]
    for name,result in outcomes_for(raw,raw_labels,raw_images).items():
        image_mean_ci(export,dict(context,space='raw',analysis=name),'pair_cosine',
                      np.asarray(result),raw_images,'common_mode_stability')
    if origin is None:
        origin=raw.mean(axis=0) if len(raw) else np.zeros(raw.shape[1])
    centered,centered_valid=unit(raw-origin)
    centered_labels=raw_labels[centered_valid]; centered_images=raw_images[centered_valid]
    for name,result in outcomes_for(centered,centered_labels,centered_images).items():
        image_mean_ci(export,dict(context,space='cmc',analysis=name),'pair_cosine',
                      np.asarray(result),centered_images,'common_mode_stability')


def alignment(export,root,artifact,table,model,dataset,layer,delta,roles,labels,perts,images):
    """Pair class/source-image means across perturbations on confirmation only."""
    fit_size_limit=export.config['protocol']['primary_fit_size']

    def _object_id(index):
        # Layerwise metadata rows are mappings; FeatureTable rows are objects.
        item=table.metadata[index]
        return item['object_id'] if isinstance(item,dict) else item.object_id

    def _limit_fit(indices):
        """Deterministic per-class fit budget in the runner's object-id hash order."""
        if not fit_size_limit or fit_size_limit<=0:
            return np.asarray(indices,dtype=np.int64)
        chosen=[]
        for class_id in sorted(set(labels[indices].tolist())):
            class_indices=indices[labels[indices]==class_id]
            ordered=sorted(class_indices.tolist(),
                           key=lambda index:(hashlib.sha256(str(_object_id(index)).encode('utf-8')).hexdigest(),
                                             str(_object_id(index))))
            chosen.extend(ordered[:fit_size_limit])
        return np.asarray(sorted(chosen),dtype=np.int64)

    def summaries_for(space):
        summaries={}
        for pert in np.unique(perts):
            idx=np.flatnonzero((roles=='confirmation')&(perts==pert))
            vals,valid=unit(delta[idx]); idx=idx[valid]
            if space=='cmc':
                fit=np.flatnonzero((roles=='direction_fit')&(perts==pert))
                # Apply the same per-class fit budget the runner uses so the
                # CMC origin is not estimated from a larger direction-fit set.
                fit=_limit_fit(fit)
                fit_unit,fit_valid=unit(delta[fit])
                origin=m1_origin(fit_unit,labels[fit][fit_valid])
                if origin is None:
                    continue
                vals,valid=unit(vals-origin); idx=idx[valid]
            for c in np.unique(labels[idx]):
                for im in np.unique(images[idx][labels[idx]==c]):
                    v=vals[(labels[idx]==c)&(images[idx]==im)].mean(axis=0)
                    norm=np.linalg.norm(v)
                    if norm>1e-8: summaries[(str(pert),int(c),str(im))]=v/norm
        return summaries
    for space in ('raw','cmc'):
        summaries=summaries_for(space)
        for a,b in itertools.combinations(sorted({str(p) for p in perts}),2):
            paired=[(c,im) for p,c,im in summaries if p==a and (b,c,im) in summaries]
            if not paired: continue
            unit_images=np.asarray([im for c,im in paired]); values=np.asarray([summaries[(a,c,im)]@summaries[(b,c,im)] for c,im in paired])
            im_values=np.asarray([values[unit_images==im].mean() for im in np.unique(unit_images)])
            draws=bootstrap_mean_samples(im_values,export.bootstrap_replicates,export.config['split']['seed'])
            lo,hi=np.quantile(draws,[.025,.975]) if len(draws) else (np.nan,np.nan)
            context=export.context(root,'alignment_recomputed',{'model_id': model,'dataset_id': dataset,'layer': layer,'source_perturbation': a,'target_perturbation': b},artifact)
            context.update(space=space, aggregation='object -> class/image direction -> paired cosine -> source image -> mean',
                           n_source_images=len(im_values),valid_n_objects=None,analysis='matched_class_source_image_cross_perturbation',
                           bootstrap_valid_replicates=len(draws),bootstrap_replicates=len(draws))
            export.emit('layerwise_alignment',context,'cross_perturbation_cosine',float(im_values.mean()),float(lo),float(hi))
            export.units.write(json.dumps({'context': context,'source_image': unit_images.tolist(),'class_id': [c for c,im in paired],'raw_value': values.tolist()},default=str)+'\n')


def _class_image_means(scores, labels, images, classes):
    """Mean score row per class averaged over its source-image cell means.

    ``flatnonzero`` keeps the original row order inside every cell, so the
    element sequence fed to ``np.mean`` is identical to the historical
    ``scores[mask & (images == image)]`` selection.
    """
    means=[]
    for c in classes:
        rows=np.flatnonzero(labels==c)
        class_images=images[rows]
        means.append(np.stack([scores[rows[class_images==image]].mean(axis=0)
            for image in np.unique(class_images)]).mean(axis=0))
    return np.stack(means)


def classwise(export, context, scores, labels, images, classes):
    for j,c in enumerate(classes):
        rows=np.flatnonzero(labels==c)
        if not len(rows): continue
        class_images=images[rows]
        means=np.stack([scores[rows[class_images==image]].mean(axis=0)
            for image in np.unique(class_images)]).mean(axis=0)
        wrong=means.copy(); wrong[j]=-np.inf; wi=int(np.argmax(wrong))
        ctx=dict(context,class_id=int(c),valid_n_objects=len(rows),n_source_images=len(np.unique(class_images)),
                 strongest_wrong_class=int(classes[wi]),aggregation='object -> class/source-image mean -> class mean')
        for metric,value in [('diagonal_similarity',means[j]),('diagonal_margin',means[j]-means[wi]),
                             ('own_class_win',int(means[j]>=means.max())),('ovr_auc',binary_auc((labels==c).astype(int),scores[:,j]))]:
            export.emit('specificity_classwise',ctx,metric,float(value))


def mapping_permutation(export, context, scores, labels, images, classes):
    """Class-direction correspondence null; permute entire direction labels.

    This preserves source-image dependence and is explicitly not an object
    label shuffle. A single permutation is shared by all image/class cells.
    """
    if any(not np.any(labels==c) for c in classes):
        return  # Missing confirmation classes do not define a full bijection null.
    means=_class_image_means(scores,labels,images,classes)
    n=len(classes); rng=np.random.default_rng(context['split_seed'])
    auc=np.array([[binary_auc((labels==c).astype(int),scores[:,j]) for j in range(n)] for c in classes])
    row_index=np.arange(n); row_max=means.max(axis=1)
    finite_means=bool(np.isfinite(means).all())
    if finite_means and n>1:
        row_second=means[row_index,np.argsort(means,axis=1)[:,n-2]]
    else:
        row_second=np.full(n,-np.inf)
    def metrics_batch(orders):
        own=means[row_index[None,:],orders]
        if finite_means:
            # Removing the selected column's own value leaves the row maximum
            # unless that value was a maximizer, in which case the second
            # largest value is the remaining maximum (ties included).
            wrong=np.where(own>=row_max[None,:],row_second[None,:],row_max[None,:])
        else:
            wrong=np.where(orders[:,:,None]==row_index[None,None,:],-np.inf,means[None,:,:]).max(axis=2)
        return np.stack([np.mean(own>=row_max,axis=1),np.mean(own-wrong,axis=1),
                         np.mean(auc[row_index[None,:],orders],axis=1)],axis=1)
    observed=metrics_batch(row_index[None,:])[0]
    replicates=int(export.permutation_replicates)
    exceed=np.zeros(3,dtype=np.int64)
    # Permutations are still drawn one at a time on the caller RNG (identical
    # sequence to the historical loop); only the metric reduction is batched.
    for start in range(0,replicates,2048):
        size=min(2048,replicates-start)
        orders=np.empty((size,n),dtype=np.int64)
        for i in range(size): orders[i]=rng.permutation(n)
        exceed+=np.count_nonzero(metrics_batch(orders)>=observed-1e-12,axis=0)
    p=(exceed+1)/(replicates+1)
    for metric,value in zip(['own_argmax_rate','mean_diagonal_margin','macro_ovr_auc'],p):
        for row in reversed(export.master):
            if all(row.get(k)==context.get(k) for k in ['run_id','stage','dataset','model','layer','perturbation','operator','normalization']) and row['metric']==metric:
                row.update(p_value=float(value),permutation_replicates=export.permutation_replicates,
                           family=f"{context['stage']}:{metric}:class_direction_correspondence",
                           permutation_null='uniform bijection of class-to-direction labels; one-sided',
                           statistics_status='cluster_bootstrap_and_class_mapping_permutation')
                break


_READOUT_DROP_METRICS=('linear_accuracy_drop','linear_margin_drop',
                       'prototype_accuracy_drop','prototype_margin_drop')


def _split_readout_alpha(key):
    condition,_,alpha=str(key).partition('@')
    return condition,(alpha or None)


def export_layerwise_readout(export,root):
    """Layerwise centered-space readout intervention from persisted E06 metrics.

    ``run_e06_readout`` records per layer/perturbation/condition/alpha drop
    metrics under ``readout_intervention``.  Metric names are the payload's own
    keys; payloads without the block are emitted as explicit missing evidence
    rather than skipped, and nothing is imputed.
    """
    for path in sorted((root/'metrics').glob('E06_*.json')):
        data=read_json(path)
        model=data.get('model_id'); dataset=data.get('dataset_id')
        intervention=data.get('readout_intervention')
        if not isinstance(intervention,dict):
            export.missing('layerwise_readout','E06 readout_intervention payload absent',
                           model=model,dataset=dataset,run_id=root.name)
            continue
        export.artifact(path)
        for layer,record in (intervention.get('layers') or {}).items():
            for perturbation,result in (record.get('by_perturbation') or {}).items():
                if not isinstance(result,dict):
                    continue
                for alpha_key,drops in (result.get('alpha') or {}).items():
                    if not isinstance(drops,dict):
                        continue
                    condition,alpha=_split_readout_alpha(alpha_key)
                    item={'model_id': model,'dataset_id': dataset,'layer': layer,'perturbation': perturbation,
                              'condition': condition,'alpha': alpha,'space': 'centered'}
                    context=export.context(root,'E06_readout',item,path)
                    context.update(valid_n_objects=result.get('confirmation_objects'),
                                   analysis='centered_readout_intervention')
                    for metric in _READOUT_DROP_METRICS:
                        if metric in drops:
                            export.emit('layerwise_readout',context,metric,drops[metric])
            trend=(((intervention.get('summary') or {})
                    .get('own_minus_wrong_linear_margin_drop_by_layer')) or {}).get(
                        'own_minus_wrong_linear_margin_drop') or {}
            if layer in trend:
                item={'model_id': model,'dataset_id': dataset,'layer': layer,'space': 'centered'}
                context=export.context(root,'E06_readout',item,path)
                context.update(analysis='own_minus_wrong_linear_margin_drop')
                export.emit('layerwise_readout',context,'own_minus_wrong_linear_margin_drop',trend[layer])
