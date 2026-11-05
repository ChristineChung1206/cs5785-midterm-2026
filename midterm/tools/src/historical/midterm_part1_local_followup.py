"""Bounded local-ensemble, Other-kernel, and expected-sample-F1 study.

The fixed reference is the exact kernel recipe that scored 0.88568 publicly.
Selection is nested; the recorded plan cannot change in an existing output dir.
"""
from __future__ import annotations

from _historical_support import prepare_output
import argparse
from dataclasses import dataclass
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
from midterm_evaluation import REASONS, dataset_path

EMB = previous.EMB
EPS = 1e-12
STAGES = ('baseline', 'ensemble', 'other', 'final')
REFERENCE_RUN = 'part1_local_consensus_2026_11_02'
REFERENCE_FILE = 'submission.csv'
REFERENCE_SHA = '6a6949390a85a5d4e767672fc718b2727b9633094574f4c1fcd90c5fee55487c'
BASE_CONFIG = dict(geometry='global', family='ridge', bandwidth=.5, strength=.1)
CONFIGS = {
    'baseline': BASE_CONFIG,
    'cosine': dict(geometry='unit', family='ridge', bandwidth=.5, strength=.1),
    'svm': dict(geometry='global', family='svm', bandwidth=.5, strength=.3),
    'other_narrow': dict(geometry='global', family='ridge', bandwidth=2., strength=.1),
    'other_mixed': dict(geometry='global', family='mixed_ridge', bandwidth=[.5, 2.], strength=.1),
}
ENSEMBLES = [('baseline', 'cosine'), ('baseline', 'svm'), ('baseline', 'cosine', 'svm')]
PLAN = {
    'version': 1, 'reference_public_score': .88568, 'reference_submission_sha256': REFERENCE_SHA,
    'fixed_reference': BASE_CONFIG, 'threshold': .5,
    'outer_folds': 4, 'repeat_seeds': list(previous.SEEDS), 'inner_folds': 3,
    'grouping': 'same <=1e-4 raw-distance connected components as previous experiment',
    'scope': 'held groups excluded from gold and unlabeled feature scopes; test used only for parity and final prediction',
    'calibration': 'three grouped raw-score cross-fits, sigmoid C=1; matching seeds for all components',
    'components': CONFIGS, 'equal_probability_ensembles': ENSEMBLES,
    'component_eligibility': {'improved_inner_groups_minimum': 2, 'F1_deficit_minimum': -.02},
    'stage2': 'replace ONLY Other with narrow or 50:50 mixed RBF kernel; four columns frozen',
    'stage3': 'threshold .5+argmax versus exact expected sample-F1 under independent Bernoulli marginals; 32 truth / 31 action states',
    'selection': 'best eligible candidate within each stage by inner OOF whole-sample F1, then require gain > .002 over incoming stage; ties keep first listed option',
    'candidate_gate': {'mean_oof_gain_minimum': .005, 'positive_repeats_minimum': 2,
                       'worst_repeat_gain_minimum': -.005, 'source_gain_minimum': -.01,
                       'late56_gain_minimum': -.01, 'historical_validation_gain_minimum': -.01},
    'export': 'only final sequential procedure may be exported if all gates pass and final test decisions differ; no outer-selected stage fallback',
    'caveat': 'All 235 labels and prior experiment/public results informed this study. OOF is resampling evidence, not a fresh holdout.',
}


def mixed_rbf_kernel(a, b, gammas):
    """Convex kernel mean, with unit diagonal on its training square matrix."""
    dist = cdist(a, b, 'sqeuclidean')
    return np.mean([np.exp(-g * dist) for g in gammas], axis=0)


@dataclass
class MixedRidge:
    geometry: previous.Geometry
    train_x: np.ndarray
    gammas: tuple
    coefficients: np.ndarray

    def raw(self, x):
        return mixed_rbf_kernel(self.geometry.transform(x), self.train_x, self.gammas) @ self.coefficients


def fit_base(scope, config):
    if config['family'] != 'mixed_ridge':
        return previous.fit_base(scope, config)
    geom = previous.Geometry.fit(scope.data.x[scope.eligible], 'global')
    z = geom.transform(scope.data.x[scope.gold])
    distances = pdist(z, 'sqeuclidean')
    nonzero = distances[distances > 1e-12]
    median = float(np.median(nonzero)) if len(nonzero) else 1.
    gammas = tuple(b / median for b in config['bandwidth'])
    kernel = mixed_rbf_kernel(z, z, gammas)
    coef = solve(kernel + config['strength'] * np.eye(len(z)), scope.data.y[scope.gold], assume_a='pos')
    return MixedRidge(geom, z, gammas, coef)


def fit_calibrated(scope, config, seed):
    raw = np.full((len(scope.gold), 5), np.nan)
    members = []
    for held, inner in scope.folds(seed):
        model = fit_base(inner, config)
        raw[held] = model.raw(scope.data.x[scope.gold[held]])
        members.append(model)
    if not np.isfinite(raw).all():
        raise ValueError('Incomplete grouped calibration predictions')
    return previous.Calibrated(fit_base(scope, config), previous.Sigmoid.fit(raw, scope.data.y[scope.gold]), members)


TRUTH_SETS = np.array([[(mask >> j) & 1 for j in range(5)] for mask in range(32)], dtype=int)
ACTION_SETS = np.array(sorted(TRUTH_SETS[1:].tolist(), key=lambda row: (sum(row), sum(int(v) << j for j, v in enumerate(row)))), dtype=int)
F1_PAYOFF = 2 * (ACTION_SETS @ TRUTH_SETS.T) / (ACTION_SETS.sum(1)[:, None] + TRUTH_SETS.sum(1)[None, :])


def expected_f1_decisions(probabilities):
    """Bayes action for the SPECIFIED independence approximation, not general joint truth."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite probabilities with shape (n, 5), in [0, 1]')
    joint = np.prod(np.where(TRUTH_SETS[None, :, :], p[:, None, :], 1 - p[:, None, :]), axis=2)
    utility = joint @ F1_PAYOFF.T
    return ACTION_SETS[np.argmax(utility, axis=1)].copy()


def replace_other(probabilities, replacement):
    p = np.asarray(probabilities, float).copy()
    replacement = np.asarray(replacement, float)
    p[:, REASONS.index('Other')] = replacement
    return p


def decode(p, decoder):
    if decoder == 'threshold':
        return previous.decide(p, .5)
    if decoder == 'expected_f1':
        return expected_f1_decisions(p)
    raise ValueError(decoder)


def blend(probabilities, members):
    return np.mean([probabilities[name] for name in members], axis=0)


def score(y, p, decoder='threshold'):
    return float(previous.row_f1(y, decode(p, decoder)).mean())


def eligibility(y, baseline, other, groups):
    delta = previous.row_f1(y, decode(other, 'threshold')) - previous.row_f1(y, decode(baseline, 'threshold'))
    by_group = pd.DataFrame({'group': groups, 'delta': delta}).groupby('group').delta.mean()
    info = {'improved_groups': int((by_group > EPS).sum()), 'worsened_groups': int((by_group < -EPS).sum()),
            'sample_f1_delta': float(delta.mean())}
    return info['improved_groups'] >= 2 and info['sample_f1_delta'] >= -.02 - EPS, info


@dataclass
class FittedFollowup:
    models: dict
    recipes: dict
    selection: dict

    def probabilities(self, x):
        components = {name: model.predict(x) for name, model in self.models.items()}
        result = {}
        for stage, recipe in self.recipes.items():
            p = blend(components, recipe['members'])
            if recipe['other'] is not None:
                p = replace_other(p, components[recipe['other']][:, REASONS.index('Other')])
            result[stage] = p
        return result

    def predict_sets(self, x):
        return {stage: decode(p, self.recipes[stage]['decoder']) for stage, p in self.probabilities(x).items()}


def fit_procedure(scope, seed):
    y = scope.data.y[scope.gold]
    folds = list(scope.folds(seed))
    inner_predictions = {}
    for name, config in CONFIGS.items():
        p = np.full((len(y), 5), np.nan)
        for k, (held, inner) in enumerate(folds):
            model = fit_calibrated(inner, config, seed + 100 + k)
            p[held] = model.predict(scope.data.x[scope.gold[held]])
        inner_predictions[name] = p
    baseline = inner_predictions['baseline']
    recipes = {'baseline': {'members': ['baseline'], 'other': None, 'decoder': 'threshold'}}
    inner_scores = {'baseline': score(y, baseline)}
    candidate_records, complement = [], {}
    permitted = set()
    for name in ('cosine', 'svm'):
        allow, details = eligibility(y, baseline, inner_predictions[name], scope.data.groups[scope.gold])
        complement[name] = dict(eligible=bool(allow), **details)
        if allow:
            permitted.add(name)
    best_recipe, best_score, best_p = recipes['baseline'].copy(), inner_scores['baseline'], baseline
    for members in ENSEMBLES:
        p = blend(inner_predictions, members)
        value = score(y, p)
        allowed = set(members).difference({'baseline'}).issubset(permitted)
        candidate_records.append({'stage': 'ensemble', 'members': members, 'eligible': allowed, 'f1': value})
        if allowed and value > best_score + EPS:
            best_recipe, best_score, best_p = {'members': list(members), 'other': None, 'decoder': 'threshold'}, value, p
    if best_score <= inner_scores['baseline'] + .002 + EPS:
        best_recipe, best_score, best_p = recipes['baseline'].copy(), inner_scores['baseline'], baseline
    recipes['ensemble'], inner_scores['ensemble'] = best_recipe.copy(), best_score
    stage1_p = best_p.copy()
    for name in ('other_narrow', 'other_mixed'):
        p = replace_other(stage1_p, inner_predictions[name][:, REASONS.index('Other')])
        value = score(y, p)
        candidate_records.append({'stage': 'other', 'model': name, 'f1': value})
        if value > best_score + EPS:
            best_recipe, best_score, best_p = {**recipes['ensemble'], 'other': name}, value, p
    if best_score <= inner_scores['ensemble'] + .002 + EPS:
        best_recipe, best_score, best_p = recipes['ensemble'].copy(), inner_scores['ensemble'], stage1_p
    recipes['other'], inner_scores['other'] = best_recipe.copy(), best_score
    expected_score = score(y, best_p, 'expected_f1')
    candidate_records.append({'stage': 'decoder', 'decoder': 'expected_f1', 'f1': expected_score})
    if expected_score > best_score + .002 + EPS:
        best_recipe, best_score = {**best_recipe, 'decoder': 'expected_f1'}, expected_score
    recipes['final'], inner_scores['final'] = best_recipe.copy(), best_score
    # Matching seed across all components is essential to isolate the model change.
    models = {name: fit_calibrated(scope, config, seed + 900) for name, config in CONFIGS.items()}
    selection = {'recipes': recipes, 'inner_scores': inner_scores, 'candidates': candidate_records,
                 'component_complementarity': complement,
                 'fit_gold_ids': scope.data.frame.iloc[scope.gold].ID.tolist(),
                 'fit_feature_ids': scope.data.frame.iloc[scope.eligible].ID.tolist()}
    return FittedFollowup(models, recipes, selection)


def prediction_frame(data, held, model, evaluation, repeat=-1, fold=-1):
    frame = data.frame.iloc[held][['ID', 'source', 'start_year']].reset_index(drop=True).copy()
    frame['group'], frame['evaluation'], frame['repeat'], frame['fold'] = data.groups[held], evaluation, repeat, fold
    y = data.y[held]
    for j, reason in enumerate(REASONS):
        frame['true_' + reason] = y[:, j]
    for stage, p in model.probabilities(data.x[held]).items():
        pred = decode(p, model.recipes[stage]['decoder'])
        frame[stage + '_f1'] = previous.row_f1(y, pred)
        frame[stage + '_exact'] = (y == pred).all(1)
        frame[stage + '_set'] = [' '.join(r for r, bit in zip(REASONS, row) if bit) for row in pred]
        for j, reason in enumerate(REASONS):
            frame[f'{stage}_prob_{reason}'] = p[:, j]
            frame[f'{stage}_pred_{reason}'] = pred[:, j]
    return frame


def bootstrap(frame):
    grouped = frame.assign(delta=frame.final_f1-frame.baseline_f1).groupby('group').delta.agg(['sum', 'count'])
    samples = np.random.default_rng(42).integers(len(grouped), size=(2000, len(grouped)))
    deltas = grouped['sum'].to_numpy()[samples].sum(1) / grouped['count'].to_numpy()[samples].sum(1)
    return np.quantile(deltas, [.025, .975]).tolist()


def run(root, output):
    root, out = prepare_output(root, output)
    # JSON round-trip turns tuples into lists for a stable repeated-run comparison.
    plan = json.loads(json.dumps(PLAN))
    if (out/'experiment_plan.json').exists() and json.loads((out/'experiment_plan.json').read_text()) != plan:
        raise ValueError('Recorded plan differs; use a new output directory')
    previous.write_json(out/'experiment_plan.json', plan)
    before = previous.protected_hashes(root)
    ref = root/'models/part1/03_kernel_ridge'
    for name in (REFERENCE_FILE, 'model.joblib'):
        before[str(ref/name)] = previous.digest(ref/name)
    previous.write_json(out/'protected_before.json', before)
    data, test = previous.load_data(root)
    full_scope = data.scope(data.gold, np.arange(len(data.frame)))
    ref_model = joblib.load(ref/'model.joblib')
    p_saved = ref_model.predict(test[EMB].to_numpy())
    p_refit = fit_calibrated(full_scope, BASE_CONFIG, 942).predict(test[EMB].to_numpy())
    gap = float(np.max(np.abs(p_saved-p_refit)))
    if gap > 1e-12 or previous.digest(ref/REFERENCE_FILE) != REFERENCE_SHA:
        raise ValueError(f'Fixed reference parity failed: {gap}')
    print(f'Fixed .88568 reference parity maximum gap: {gap:.3g}', flush=True)
    previous.write_json(out/'metadata.json', {'source_sha256': previous.digest(__file__),
        'dependency_sha256': previous.digest(Path(previous.__file__)),
        'data_hashes': {s: previous.digest(dataset_path(root, 'part1', s)) for s in ('train', 'val', 'test')},
        'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn')},
        'reference_probability_gap': gap, 'gold_rows': len(data.gold), 'gold_groups': len(np.unique(data.groups[data.gold]))})
    pieces, selections = [], []
    started = time.time()
    for repeat, seed in enumerate(previous.SEEDS):
        for fold, (held, scope) in enumerate(full_scope.folds(seed, n=4)):
            fitted = fit_procedure(scope, seed + fold * 17)
            frame = prediction_frame(data, data.gold[held], fitted, 'grouped_oof', repeat, fold)
            pieces.append(frame)
            selections.append(dict(repeat=repeat, fold=fold, **fitted.selection))
            print(f'repeat {repeat+1} fold {fold+1}: {json.dumps(previous.summarize(frame))}', flush=True)
    oof = pd.concat(pieces, ignore_index=True)
    assert len(oof) == 705 and oof.groupby(['repeat','ID']).size().eq(1).all()
    oof.to_csv(out/'oof_predictions.csv', index=False)
    previous.write_json(out/'fold_selections.json', selections)
    source, years = data.frame.source.to_numpy(), data.frame.start_year.to_numpy()
    specs = {
        'historical_validation': (np.flatnonzero(source=='train'), np.flatnonzero(source!='validation'), np.flatnonzero(source=='validation')),
        'late_train_20': (np.flatnonzero((source=='train')&(years<=2006)), np.flatnonzero((source!='validation')&(years<=2006)), np.flatnonzero((source=='train')&(years>=2007)&(years<=2009))),
        'late_combined_56': (np.flatnonzero((source!='unlabeled')&(years<=2006)), np.flatnonzero(years<=2006), np.flatnonzero((source!='unlabeled')&(years>=2007)&(years<=2009))),
    }
    diagnostics, tables = {}, [oof]
    for name, (gold, eligible, held) in specs.items():
        model = fit_procedure(data.scope(gold, eligible), 42)
        frame = prediction_frame(data, held, model, name)
        frame['near_early_gold_match'] = np.isin(data.groups[held], data.groups[gold])
        frame.to_csv(out/f'{name}_predictions.csv', index=False)
        diagnostics[name] = {'metrics': previous.summarize(frame), 'novel_group_metrics': previous.summarize(frame.loc[~frame.near_early_gold_match]),
                             'gold_rows': len(gold), 'rows': len(held), 'selection': model.selection}
        tables.append(frame)
        print(name + ': ' + json.dumps(diagnostics[name]['metrics']), flush=True)
    previous.write_json(out/'diagnostics.json', diagnostics)
    table = pd.concat(tables, ignore_index=True)
    changed = table.loc[table.final_set != table.baseline_set].copy()
    changed['sample_f1_delta'] = changed.final_f1-changed.baseline_f1
    changed.to_csv(out/'changed_trials.csv', index=False)
    reason_records, year_records, set_records = [], [], []
    for name, frame in table.groupby('evaluation'):
        for stage in STAGES:
            for reason in REASONS:
                y, p = frame['true_'+reason].to_numpy(), frame[f'{stage}_pred_{reason}'].to_numpy()
                tp, fp, fn = int((y*p).sum()), int(((1-y)*p).sum()), int((y*(1-p)).sum())
                reason_records.append(dict(evaluation=name, stage=stage, reason=reason, tp=tp, fp=fp, fn=fn,
                                           f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.))
        for year, sub in frame.groupby('start_year'):
            year_records.append(dict(evaluation=name, year=int(year), observations=len(sub), **{m:v['sample_f1'] for m,v in previous.summarize(sub).items()}))
        true_sets = frame[['true_'+r for r in REASONS]].apply(lambda row:' '.join(r for r,v in zip(REASONS,row) if v), axis=1)
        for true_set, sub in frame.groupby(true_sets):
            set_records.append(dict(evaluation=name, true_set=true_set, unique_trials=sub.ID.nunique(), observations=len(sub),
                                    **{m:v['sample_f1'] for m,v in previous.summarize(sub).items()}))
    pd.DataFrame(reason_records).to_csv(out/'metrics_by_reason.csv',index=False)
    pd.DataFrame(year_records).to_csv(out/'metrics_by_year.csv',index=False)
    pd.DataFrame(set_records).to_csv(out/'metrics_by_true_set.csv',index=False)
    overall = previous.summarize(oof)
    repeats = {str(k):previous.summarize(v) for k,v in oof.groupby('repeat')}
    sources = {str(k):previous.summarize(v) for k,v in oof.groupby('source')}
    delta = lambda scores:scores['final']['sample_f1']-scores['baseline']['sample_f1']
    gains = [delta(s) for s in repeats.values()]
    gates = {'mean_oof_gain':delta(overall)>=.005-EPS, 'positive_repeats':sum(g>EPS for g in gains)>=2,
             'worst_repeat':min(gains)>=-.005-EPS, 'both_sources':all(delta(s)>=-.01-EPS for s in sources.values()),
             'late56':delta(diagnostics['late_combined_56']['metrics'])>=-.01-EPS,
             'historical_validation':delta(diagnostics['historical_validation']['metrics'])>=-.01-EPS}
    summary = {'overall':overall, 'by_repeat':repeats, 'by_source':sources, 'repeat_gains':gains,
               'mean_gain':delta(overall), 'group_bootstrap_95_percent':bootstrap(oof),
               'diagnostic_metrics':{k:v['metrics'] for k,v in diagnostics.items()},
               'candidate_gate':gates, 'candidate_allowed':all(gates.values()), 'candidate':None,
               'selection_counts':{stage:{key:sum(json.dumps(s['recipes'][stage],sort_keys=True)==key for s in selections)
                                          for key in sorted({json.dumps(s['recipes'][stage],sort_keys=True) for s in selections})} for stage in STAGES},
               'elapsed_seconds':time.time()-started, 'caveat':PLAN['caveat']}
    if all(gates.values()):
        fitted = fit_procedure(full_scope,42)
        predictions = fitted.predict_sets(test[EMB].to_numpy())
        changed_count = int((predictions['final']!=predictions['baseline']).any(1).sum())
        previous.write_json(out/'tools/archive/records/final_selection_nov01.json',fitted.selection)
        if changed_count:
            submission = pd.DataFrame({'ID':test.ID,'Predicted':[' '.join(r for r,b in zip(REASONS,row) if b) for row in predictions['final']]})
            contents = submission.to_csv(index=False).encode()
            import hashlib
            sha = hashlib.sha256(contents).hexdigest()
            name = f'part1_local_followup_{sha[:12]}.csv'
            (out/name).write_bytes(contents)
            joblib.dump(fitted,out/'model.joblib')
            summary['candidate'] = {'filename':name,'sha256':sha,'model_sha256':previous.digest(out/'model.joblib'),
                                    'rows':len(submission),'changed_vs_0_88568':changed_count,'kaggle_uploaded':False}
        else:
            summary['export_note'] = 'Full-data selection produced unchanged reference predictions; no duplicate submission exported.'
    previous.write_json(out/'summary.json',summary)
    for path, sha in before.items():
        if previous.digest(path)!=sha:
            raise RuntimeError(f'Protected file changed: {path}')
    previous.write_json(out/'verification.json',{'protected_files_unchanged':True,'gold_rows':235,'oof_rows':len(oof),'outer_folds':12,
        'output_hashes':{p.name:previous.digest(p) for p in out.glob('*') if p.is_file() and p.name!='verification.json'}})
    print(json.dumps(summary,indent=2),flush=True)




if __name__ == "__main__":
    from replay import main
    main(default_study='local_followup')
