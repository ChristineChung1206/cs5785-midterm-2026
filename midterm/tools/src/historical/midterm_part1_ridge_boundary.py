"""Prespecified ridge boundary grid followed by scope-local group weighting."""
from __future__ import annotations

from _historical_support import prepare_output
import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.spatial.distance import cdist, pdist
from threadpoolctl import threadpool_limits

import midterm_part1_local_consensus as previous
import midterm_part1_local_followup as recent
from midterm_evaluation import REASONS, dataset_path

EMB = previous.EMB
EPS = 1e-12
BASE_CONFIG = dict(bandwidth=.5, strength=.1, beta=0.)
BANDWIDTHS = (.125, .25, .5, 1.)
STRENGTHS = (.01, .03, .1, .3)
BETAS = (0., .5, 1.)
GRID = [BASE_CONFIG.copy()] + [dict(bandwidth=b, strength=a, beta=0.) for b in BANDWIDTHS for a in STRENGTHS if (b,a)!=(.5,.1)]
STAGES = ('baseline', 'tuned', 'final')
PLAN = {
    'version':1, 'reference_public_score':.88568, 'reference_submission_sha256':recent.REFERENCE_SHA,
    'fixed_reference':BASE_CONFIG, 'geometry':'raw Euclidean, single global scale',
    'kernel':'exp(-b * squared_distance / median_nonzero_training_gold_squared_distance)',
    'boundary_grid':GRID, 'weight_exponents':BETAS,
    'weight_formula':'n_gold_in_current_fitting_group ** (-beta), normalized to mean 1; unlabeled and held rows do not count',
    'weighted_objective':'sum_i w_i * ||y_i-f(x_i)||^2 + alpha*||f||_H^2',
    'weighting_scope':'ridge loss only; gamma normalization and sigmoid calibration remain unweighted',
    'outer_folds':4, 'repeat_seeds':list(previous.SEEDS), 'inner_folds':3,
    'grouping':'same <=1e-4 raw Euclidean connected components; held groups removed from gold and feature pools',
    'calibration':'three grouped raw-score cross-fits; unweighted sigmoid C=1; same folds and seeds for all candidates',
    'selection':'stage1 best of16 inner OOF sample-F1; stage2 best beta for retained stage1 config; each stage needs gain >.002 over incoming stage',
    'tie_break':'reference first, then listed b/alpha order; beta0 first, then .5, then1',
    'decision':'fixed threshold .5 plus argmax for empty set; all evaluation is unweighted per trial',
    'candidate_gate':recent.PLAN['candidate_gate'],
    'export':recent.PLAN['export'],
    'caveat':recent.PLAN['caveat'],
}


def group_weights(groups, beta):
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups) or not np.isfinite(beta) or beta < 0:
        raise ValueError('Expected nonempty group vector and finite nonnegative beta')
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = counts[inverse].astype(float) ** (-beta)
    return weights / weights.mean()


def fit_base(scope, config):
    if config['beta'] == 0:
        return previous.fit_base(scope, {**recent.BASE_CONFIG, 'bandwidth':config['bandwidth'], 'strength':config['strength']})
    geom = previous.Geometry.fit(scope.data.x[scope.eligible], 'global')
    z = geom.transform(scope.data.x[scope.gold])
    distances = pdist(z, 'sqeuclidean')
    nonzero = distances[distances > 1e-12]
    median = float(np.median(nonzero)) if len(nonzero) else 1.
    gamma = config['bandwidth'] / median
    kernel = np.exp(-gamma * cdist(z, z, 'sqeuclidean'))
    weights = group_weights(scope.data.groups[scope.gold], config['beta'])
    coefficients = solve(kernel + np.diag(config['strength'] / weights), scope.data.y[scope.gold], assume_a='pos')
    return previous.BaseModel(geom, 'ridge', z, gamma, [], coefficients)


def fit_calibrated(scope, config, seed):
    raw = np.full((len(scope.gold), 5), np.nan)
    members = []
    for held, inner in scope.folds(seed):
        base = fit_base(inner, config)
        raw[held] = base.raw(scope.data.x[scope.gold[held]])
        members.append(base)
    if not np.isfinite(raw).all():
        raise ValueError('Incomplete calibration OOF scores')
    return previous.Calibrated(fit_base(scope, config), previous.Sigmoid.fit(raw, scope.data.y[scope.gold]), members)


def inner_predictions(scope, config, folds, seed):
    p = np.full((len(scope.gold), 5), np.nan)
    for k, (held, inner) in enumerate(folds):
        model = fit_calibrated(inner, config, seed + 100 + k)
        p[held] = model.predict(scope.data.x[scope.gold[held]])
    return p


@dataclass
class FittedBoundary:
    models: dict
    recipes: dict
    selection: dict

    def probabilities(self, x):
        return {stage:model.predict(x) for stage,model in self.models.items()}

    def predict(self, x):
        return self.models['final'].predict(x)

    def predict_sets(self, x):
        return {stage:previous.decide(p,.5) for stage,p in self.probabilities(x).items()}


def fit_procedure(scope, seed):
    y = scope.data.y[scope.gold]
    folds = list(scope.folds(seed))
    grid_scores, weight_scores = [], []
    best_config = BASE_CONFIG.copy()
    best_score = -np.inf
    baseline_score = None
    for config in GRID:
        p = inner_predictions(scope, config, folds, seed)
        value = previous.f1(y,p,.5)
        grid_scores.append(dict(config=config, sample_f1=value))
        if config == BASE_CONFIG:
            baseline_score = value
        if value > best_score + EPS:
            best_score, best_config = value, config.copy()
    if best_score <= baseline_score + .002 + EPS:
        best_score, best_config = baseline_score, BASE_CONFIG.copy()
    tuned_config, tuned_score = best_config.copy(), best_score
    weight_scores.append(dict(config=tuned_config.copy(), sample_f1=tuned_score))
    for beta in BETAS[1:]:
        config = {**tuned_config, 'beta':beta}
        p = inner_predictions(scope, config, folds, seed)
        value = previous.f1(y,p,.5)
        weight_scores.append(dict(config=config, sample_f1=value))
        if value > best_score + EPS:
            best_score, best_config = value, config.copy()
    if best_score <= tuned_score + .002 + EPS:
        best_score, best_config = tuned_score, tuned_config.copy()
    configs = {'baseline':BASE_CONFIG.copy(), 'tuned':tuned_config, 'final':best_config}
    recipes = {stage:{**config,'threshold':.5,'decoder':'threshold'} for stage,config in configs.items()}
    cache, models = {}, {}
    for stage,config in configs.items():
        key = json.dumps(config,sort_keys=True)
        if key not in cache:
            cache[key] = fit_calibrated(scope, config, seed+900)
        models[stage] = cache[key]
    selection = {'recipes':recipes, 'grid_scores':grid_scores, 'weight_scores':weight_scores,
        'inner_scores':{'baseline':baseline_score,'tuned':tuned_score,'final':best_score},
        'fit_gold_ids':scope.data.frame.iloc[scope.gold].ID.tolist(),
        'fit_feature_ids':scope.data.frame.iloc[scope.eligible].ID.tolist(),
        'selected_weights':group_weights(scope.data.groups[scope.gold],best_config['beta']).tolist()}
    return FittedBoundary(models,recipes,selection)


def save_breakdowns(table, out):
    reason_rows, year_rows, set_rows = [], [], []
    for evaluation, frame in table.groupby('evaluation'):
        for stage in STAGES:
            for reason in REASONS:
                y, p = frame['true_'+reason].to_numpy(), frame[f'{stage}_pred_{reason}'].to_numpy()
                tp, fp, fn = int((y*p).sum()), int(((1-y)*p).sum()), int((y*(1-p)).sum())
                reason_rows.append(dict(evaluation=evaluation,stage=stage,reason=reason,tp=tp,fp=fp,fn=fn,
                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.))
        for year, sub in frame.groupby('start_year'):
            year_rows.append(dict(evaluation=evaluation,year=int(year),observations=len(sub),
                **{s:m['sample_f1'] for s,m in previous.summarize(sub).items()}))
        true_sets=frame[['true_'+r for r in REASONS]].apply(lambda row:' '.join(r for r,v in zip(REASONS,row) if v),axis=1)
        for true_set, sub in frame.groupby(true_sets):
            set_rows.append(dict(evaluation=evaluation,true_set=true_set,unique_trials=sub.ID.nunique(),observations=len(sub),
                **{s:m['sample_f1'] for s,m in previous.summarize(sub).items()}))
    pd.DataFrame(reason_rows).to_csv(out/'metrics_by_reason.csv',index=False)
    pd.DataFrame(year_rows).to_csv(out/'metrics_by_year.csv',index=False)
    pd.DataFrame(set_rows).to_csv(out/'metrics_by_true_set.csv',index=False)


def run(root, output):
    root, out = prepare_output(root, output)
    plan=json.loads(json.dumps(PLAN))
    if (out/'experiment_plan.json').exists() and json.loads((out/'experiment_plan.json').read_text())!=plan:
        raise ValueError('Recorded plan differs; use a new output directory')
    previous.write_json(out/'experiment_plan.json',plan)
    reference=root/'models/part1/03_kernel_ridge'
    before=previous.protected_hashes(root)
    for name in (recent.REFERENCE_FILE,'model.joblib'):
        before[str(reference/name)]=previous.digest(reference/name)
    previous.write_json(out/'protected_before.json',before)
    data,test=previous.load_data(root)
    full_scope=data.scope(data.gold,np.arange(len(data.frame)))
    saved=joblib.load(reference/'model.joblib').predict(test[EMB].to_numpy())
    refit=fit_calibrated(full_scope,BASE_CONFIG,942).predict(test[EMB].to_numpy())
    gap=float(np.max(np.abs(saved-refit)))
    if gap>1e-12 or previous.digest(reference/recent.REFERENCE_FILE)!=recent.REFERENCE_SHA:
        raise ValueError('Reference parity failed')
    print(f'Fixed .88568 reference probability gap: {gap}',flush=True)
    previous.write_json(out/'metadata.json',{'source_sha256':previous.digest(__file__),
        'dependencies':{Path(m.__file__).name:previous.digest(m.__file__) for m in (previous,recent)},
        'versions':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn')},
        'data_hashes':{s:previous.digest(dataset_path(root,'part1',s)) for s in ('train','val','test')},
        'reference_probability_gap':gap,'gold_rows':235,'gold_groups':len(np.unique(data.groups[data.gold]))})
    audit=data.frame.iloc[data.gold][['ID','source','start_year']].reset_index(drop=True)
    audit['group']=data.groups[data.gold]
    for beta in BETAS:
        audit[f'weight_beta_{beta}']=group_weights(data.groups[data.gold],beta)
    audit.to_csv(out/'group_weight_audit.csv',index=False)
    pieces,selections=[],[]
    started=time.time()
    for repeat,seed in enumerate(previous.SEEDS):
        for fold,(held,scope) in enumerate(full_scope.folds(seed,n=4)):
            fitted=fit_procedure(scope,seed+fold*17)
            frame=recent.prediction_frame(data,data.gold[held],fitted,'grouped_oof',repeat,fold)
            pieces.append(frame)
            selections.append(dict(repeat=repeat,fold=fold,**fitted.selection))
            print(f'repeat{repeat+1} fold{fold+1}: {json.dumps(previous.summarize(frame))}',flush=True)
    oof=pd.concat(pieces,ignore_index=True)
    if len(oof)!=705 or not oof.groupby(['repeat','ID']).size().eq(1).all():
        raise ValueError('Incomplete OOF')
    # The unchanged fixed reference must reproduce the last round on identical folds.
    last=pd.read_csv(root/'models/part1/exploration/local_followup/oof_predictions.csv').set_index(['repeat','fold','ID'])
    check=oof.set_index(['repeat','fold','ID']).loc[last.index]
    np.testing.assert_allclose(check[[f'baseline_prob_{r}' for r in REASONS]],last[[f'baseline_prob_{r}' for r in REASONS]],atol=1e-12,rtol=0)
    oof.to_csv(out/'oof_predictions.csv',index=False)
    previous.write_json(out/'fold_selections.json',selections)
    source,years=data.frame.source.to_numpy(),data.frame.start_year.to_numpy()
    specs={
        'historical_validation':(np.flatnonzero(source=='train'),np.flatnonzero(source!='validation'),np.flatnonzero(source=='validation')),
        'late_train_20':(np.flatnonzero((source=='train')&(years<=2006)),np.flatnonzero((source!='validation')&(years<=2006)),np.flatnonzero((source=='train')&(years>=2007)&(years<=2009))),
        'late_combined_56':(np.flatnonzero((source!='unlabeled')&(years<=2006)),np.flatnonzero(years<=2006),np.flatnonzero((source!='unlabeled')&(years>=2007)&(years<=2009))),
    }
    diagnostics,tables={},[oof]
    for name,(gold,eligible,held) in specs.items():
        fitted=fit_procedure(data.scope(gold,eligible),42)
        frame=recent.prediction_frame(data,held,fitted,name)
        frame['near_early_gold_match']=np.isin(data.groups[held],data.groups[gold])
        frame.to_csv(out/f'{name}_predictions.csv',index=False)
        diagnostics[name]={'metrics':previous.summarize(frame),'novel_group_metrics':previous.summarize(frame.loc[~frame.near_early_gold_match]),
            'rows':len(held),'gold_rows':len(gold),'selection':fitted.selection}
        tables.append(frame)
        print(name+': '+json.dumps(diagnostics[name]['metrics']),flush=True)
    previous.write_json(out/'diagnostics.json',diagnostics)
    table=pd.concat(tables,ignore_index=True)
    changed=table.loc[table.final_set!=table.baseline_set].copy()
    changed['sample_f1_delta']=changed.final_f1-changed.baseline_f1
    changed.to_csv(out/'changed_trials.csv',index=False)
    save_breakdowns(table,out)
    overall=previous.summarize(oof)
    repeats={str(k):previous.summarize(v) for k,v in oof.groupby('repeat')}
    sources={str(k):previous.summarize(v) for k,v in oof.groupby('source')}
    delta=lambda s:s['final']['sample_f1']-s['baseline']['sample_f1']
    gains=[delta(v) for v in repeats.values()]
    gates={'mean_oof_gain':delta(overall)>=.005-EPS,'positive_repeats':sum(g>EPS for g in gains)>=2,
        'worst_repeat':min(gains)>=-.005-EPS,'both_sources':all(delta(v)>=-.01-EPS for v in sources.values()),
        'late56':delta(diagnostics['late_combined_56']['metrics'])>=-.01-EPS,
        'historical_validation':delta(diagnostics['historical_validation']['metrics'])>=-.01-EPS}
    summary={'overall':overall,'by_repeat':repeats,'by_source':sources,'repeat_gains':gains,'mean_gain':delta(overall),
        'group_bootstrap_95_percent':recent.bootstrap(oof),'diagnostic_metrics':{k:v['metrics'] for k,v in diagnostics.items()},
        'candidate_gate':gates,'candidate_allowed':all(gates.values()),'candidate':None,
        'selection_counts':{stage:{key:sum(json.dumps(s['recipes'][stage],sort_keys=True)==key for s in selections)
            for key in sorted({json.dumps(s['recipes'][stage],sort_keys=True) for s in selections})} for stage in STAGES},
        'elapsed_seconds':time.time()-started,'caveat':PLAN['caveat']}
    if all(gates.values()):
        fitted=fit_procedure(full_scope,42)
        predictions=fitted.predict_sets(test[EMB].to_numpy())
        previous.write_json(out/'tools/archive/records/final_selection_nov01.json',fitted.selection)
        changed_count=int((predictions['final']!=predictions['baseline']).any(1).sum())
        if changed_count:
            submission=pd.DataFrame({'ID':test.ID,'Predicted':[' '.join(r for r,b in zip(REASONS,row) if b) for row in predictions['final']]})
            contents=submission.to_csv(index=False).encode()
            sha=hashlib.sha256(contents).hexdigest()
            filename=f'part1_ridge_boundary_{sha[:12]}.csv'
            (out/filename).write_bytes(contents)
            joblib.dump(fitted,out/'model.joblib')
            summary['candidate']={'filename':filename,'sha256':sha,'model_sha256':previous.digest(out/'model.joblib'),
                'rows':len(submission),'changed_vs_0_88568':changed_count,'kaggle_uploaded':False}
        else:
            summary['export_note']='Final decisions unchanged; no duplicate submission exported.'
    previous.write_json(out/'summary.json',summary)
    for path,sha in before.items():
        if previous.digest(path)!=sha:
            raise RuntimeError(f'Protected file changed: {path}')
    previous.write_json(out/'verification.json',{'protected_files_unchanged':True,'baseline_oof_matches_previous':True,
        'gold_rows':235,'oof_rows':len(oof),'outer_folds':12,
        'output_hashes':{p.name:previous.digest(p) for p in out.glob('*') if p.is_file() and p.name!='verification.json'}})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    from midterm_part1_ridge_boundary import main as run_main
    with threadpool_limits(limits=1):
        run_main()


if __name__ == "__main__":
    from replay import main
    main(default_study='ridge_boundary')
