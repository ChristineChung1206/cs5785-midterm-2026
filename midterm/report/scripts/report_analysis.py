"""Recompute report tables/figures from retained predictions; never fits a model."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "tools" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from midterm_evaluation import REASONS, evaluate_part1, evaluate_part2
from midterm_part2_baseline import analyze_failure_reasons

SOURCES = {
    'p1_supervised': 'models/part1/01_supervised_lr/evidence/original_run/validation_predictions.csv',
    'p1_initial_kmeans': 'models/part1/02_kmeans_lr/evidence/initial_run/validation_predictions.csv',
    'p1_retained_kmeans': 'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv',
    'p1_fixed_ridge': 'models/part1/exploration/local_followup/historical_validation_predictions.csv',
    'p2_baseline': 'models/part2/01_tfidf_lr/evidence/analysis/validation_predictions.csv',
    'p2_phase_shared': 'models/part2/03_phase_shared/evidence/selection/validation_candidate/validation_predictions.csv',
    'historical_reason_lookup': 'models/part1/02_kmeans_lr/evidence/selected/reason_lookup.csv',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def part1_tables(root):
    summary, by_reason, predictions = [], [], []
    truth_reference = None
    specifications = [('supervised_lr', 'p1_supervised', .5, 'prob_'),
                      ('initial_kmeans_lr', 'p1_initial_kmeans', .5, 'prob_'),
                      ('retained_kmeans_lr', 'p1_retained_kmeans', .35, 'prob_'),
                      ('fixed_ridge_157_gold', 'p1_fixed_ridge', .5, 'baseline_prob_')]
    for model, source, threshold, prefix in specifications:
        frame = pd.read_csv(root / SOURCES[source])
        reference = frame[['ID', *['true_' + r for r in REASONS]]].rename(columns={'true_' + r: r for r in REASONS})
        probabilities = frame[['ID', *[prefix + r for r in REASONS]]].rename(columns={prefix + r: r for r in REASONS})
        if truth_reference is None:
            truth_reference = reference.set_index('ID').sort_index()
        else:
            pd.testing.assert_frame_equal(reference.set_index('ID').sort_index(), truth_reference)
        result = evaluate_part1(reference, probabilities, threshold, empty_set_policy='argmax')
        y = result.predictions[['true_' + r for r in REASONS]].to_numpy()
        pred = result.predictions[['pred_' + r for r in REASONS]].to_numpy()
        summary.append(dict(model=model, threshold=threshold, gold_fit_rows=157,
                            decision_policy='threshold_then_argmax_if_empty',
                            **result.summary, exact_sets=int((pred == y).all(axis=1).sum()),
                            subset_accuracy=float((pred == y).all(axis=1).mean())))
        by_reason.append(result.by_group.assign(model=model))
        predictions.append(result.predictions.assign(model=model))
    return pd.DataFrame(summary), pd.concat(by_reason, ignore_index=True), pd.concat(predictions, ignore_index=True)


def part2_tables(root):
    lookup = pd.read_csv(root / SOURCES['historical_reason_lookup'])
    summaries, phases, analyses, predictions = [], [], {}, []
    reference_truth = None
    for model, source in [('baseline', 'p2_baseline'), ('phase_shared_best', 'p2_phase_shared')]:
        frame = pd.read_csv(root / SOURCES[source])
        reference = frame[['ID', 'trial_id', 'phase', 'label']]
        if reference_truth is None:
            reference_truth = reference.set_index('ID').sort_index()
        else:
            pd.testing.assert_frame_equal(reference.set_index('ID').sort_index(), reference_truth)
        result = evaluate_part2(reference, frame[['ID', 'p_success']], .5)
        y, p, pred = result.predictions.label, result.predictions.p_success, result.predictions.prediction
        summaries.append(dict(model=model, **result.summary,
            f1_pooled=float(f1_score(y, pred)), roc_auc_pooled=float(roc_auc_score(y, p)),
            average_precision_pooled=float(average_precision_score(y, p)),
            f1_equal_phase=float(result.by_group.f1.mean()),
            roc_auc_equal_phase=float(result.by_group.roc_auc.mean()),
            average_precision_equal_phase=float(result.by_group.average_precision.mean())))
        phases.append(result.by_group.assign(model=model))
        predictions.append(result.predictions.assign(model=model))
        analyses[model] = analyze_failure_reasons(result.predictions, lookup)
    before = analyses['baseline']['reason_rates']
    after = analyses['phase_shared_best']['reason_rates']
    keys = ['phase', 'reason_group', 'source_type']
    comparison = before.merge(after, on=keys, suffixes=('_baseline', '_best'), validate='one_to_one')
    for column in ['n_trials', 'n_observed_reasons', 'n_predicted_reasons', 'n_unmatched', 'n_training_resubstitution_reasons']:
        np.testing.assert_array_equal(comparison[column + '_baseline'], comparison[column + '_best'])
        comparison[column] = comparison[column + '_baseline']
    comparison['false_success_rate_delta'] = comparison.false_success_rate_best - comparison.false_success_rate_baseline
    failed = analyses['baseline']['failed_rows']
    provenance = {'lookup_file': SOURCES['historical_reason_lookup'],
        'lookup_sha256': sha256(root / SOURCES['historical_reason_lookup']), 'lookup_rows': len(lookup),
        'failed_trial_phase_rows': len(failed), 'unique_failed_trials': int(failed.trial_id.nunique()),
        'matched_trial_phase_rows': int(failed.matched_part1.sum()),
        'observed_reason_trial_phase_rows': int(failed.source_type.eq('observed').sum()),
        'predicted_reason_trial_phase_rows': int(failed.source_type.eq('predicted').sum()),
        'predicted_training_resubstitution_rows': int(failed.prediction_is_training_resubstitution.eq(True).sum()),
        'unmatched_trial_phase_rows': int(failed.source_type.eq('unmatched').sum()),
        'reasons_used_as_predictors': False,
        'reason_model': 'historical KMeans+LR lookup, unchanged for before/after comparison',
        'warning': 'Predicted reasons are model estimates; training resubstitution is not out-of-fold evidence. Multilabel groups overlap. Rates are descriptive, not causal.'}
    return pd.DataFrame(summaries), pd.concat(phases, ignore_index=True), analyses, comparison, provenance, pd.concat(predictions, ignore_index=True)


def make_figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = root / 'report/analysis'
    figures = root / 'report/figures'
    figures.mkdir(parents=True, exist_ok=True)
    p1 = pd.read_csv(out/'part1_validation_by_reason.csv')
    shown = ['supervised_lr', 'retained_kmeans_lr', 'fixed_ridge_157_gold']
    matrix = p1.pivot(index='reason', columns='model', values='f1').reindex(REASONS)[shown]
    ax = matrix.rename(columns={'supervised_lr':'Supervised LR', 'retained_kmeans_lr':'K-Means + LR', 'fixed_ridge_157_gold':'Fixed RBF ridge'}).plot.bar(figsize=(9,4.5), ylim=(0,1), rot=0)
    ax.set_ylabel('Validation F1 per reason'); ax.set_xlabel(''); ax.set_title('Part 1: same nonempty output policy, 78 validation trials',pad=35)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False); ax.figure.tight_layout(); ax.figure.savefig(figures/'part1_reason_f1.png', dpi=160); plt.close(ax.figure)
    p2 = pd.read_csv(out/'part2_validation_by_phase.csv')
    matrix = p2.pivot(index='phase', columns='model', values='log_loss').reindex(['I','II','III'])
    ax = matrix.rename(columns={'baseline':'TF-IDF baseline','phase_shared_best':'Phase-shared blend'}).plot.bar(figsize=(7.5,4.3), rot=0)
    ax.set_ylabel('Validation log loss (lower is better)'); ax.set_xlabel('Phase'); ax.set_title('Part 2: saved validation probabilities',pad=35)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    ax.figure.tight_layout(); ax.figure.savefig(figures/'part2_phase_logloss.png', dpi=160); plt.close(ax.figure)
    comparison = pd.read_csv(out/'part2_failure_reason_comparison.csv')
    reason_order = [*REASONS, 'no_recorded_stopping_reason']
    labels = [*REASONS, 'No Part 1 match']
    fig, axes = plt.subplots(1,2,figsize=(11.8,4.5),sharey=True)
    for ax, suffix, title in zip(axes, ['baseline','best'], ['TF-IDF baseline','Phase-shared blend']):
        selected = comparison[comparison.reason_group.isin(reason_order)]
        rates = selected.pivot(index='reason_group',columns='phase',values='false_success_rate_'+suffix).reindex(reason_order)[['I','II','III']]
        counts = selected.pivot(index='reason_group',columns='phase',values='n_trials').reindex(reason_order)[['I','II','III']]
        image = ax.imshow(rates.to_numpy(), vmin=0, vmax=1, cmap='Blues', aspect='auto')
        for i in range(len(reason_order)):
            for j in range(3):
                value, count = rates.iloc[i,j], int(counts.iloc[i,j])
                text = 'NA (n=0)' if pd.isna(value) else f'{value:.2f}\n(n={count})'
                ax.text(j,i,text,ha='center',va='center',fontsize=9,color='white' if value>.6 else 'black')
        ax.set_xticks(range(3), ['I','II','III']); ax.set_yticks(range(6), labels); ax.set_title(title); ax.set_xlabel('Phase')
    fig.suptitle('False-success rate among failed validation trials; historical reason groups')
    fig.subplots_adjust(left=.18,right=.88,top=.84,bottom=.12,wspace=.12)
    fig.colorbar(image,ax=axes,location='right',fraction=.035,pad=.035,label='False-success rate')
    fig.savefig(figures/'part2_failure_reason_comparison.png',dpi=160); plt.close(fig)
    return sorted(str(p.relative_to(root)) for p in figures.glob('*.png'))


def build_report(root=None, figures=True):
    root = Path(root) if root is not None else PROJECT_ROOT
    out = root / 'report/analysis'; out.mkdir(parents=True, exist_ok=True)
    selection_path = root / 'models/selection.json'
    selection = json.loads(selection_path.read_text())
    public_models = {part: selection[part]['best_public'] for part in ('part1', 'part2')}
    for item in public_models.values():
        submission = root / item['submission']
        if sha256(submission) != item['sha256']:
            raise ValueError(f'Selection submission checksum mismatch: {submission}')
    p1, p1_reasons, p1_predictions = part1_tables(root)
    p2, p2_phases, analyses, comparison, provenance, p2_predictions = part2_tables(root)
    tables = {'part1_validation_summary':p1, 'part1_validation_by_reason':p1_reasons,
              'part1_validation_predictions':p1_predictions, 'part2_validation_summary':p2,
              'part2_validation_by_phase':p2_phases, 'part2_validation_predictions':p2_predictions,
              'part2_failure_reason_comparison':comparison}
    for key in ['reason_rates','source_rates','source_detail_rates','overall_failed_rates','failed_rows']:
        tables['part2_failure_' + key] = pd.concat([value[key].assign(model=model) for model,value in analyses.items()],ignore_index=True)
    for name, table in tables.items():
        table.to_csv(out/(name+'.csv'),index=False)
    report = {'selection': {'path': 'models/selection.json', 'sha256': sha256(selection_path),
                             'public_models': public_models},
        'part1_validation':p1.to_dict('records'), 'part1_validation_by_reason':p1_reasons.to_dict('records'),
        'part2_validation':p2.to_dict('records'), 'part2_validation_by_phase':p2_phases.to_dict('records'),
        'part2_failure_reason_comparison':comparison.to_dict('records'), 'part2_reason_provenance':provenance,
        'sources':{name:{'path':path,'sha256':sha256(root/path)} for name,path in SOURCES.items()},
        'limitations':['All reported validation labels were previously inspected during development; held-out fitting scope does not imply an untouched test.',
          'Part1 fixed ridge validation is the train157 fit saved as baseline in local_followup, not the final235-label submission model.',
          'Part1 original supervised raw score .8654 used empty sets; fair argmax score is recomputed here.',
          'Part2 primary validation log loss gives equal weight to each phase; pooled metrics are reported separately.',
          'Reason analysis freezes the historical Part1 lookup for both Part2 models and never feeds reasons into outcome predictors.'],
        'no_model_training_performed':True}
    report['figures'] = make_figures(root) if figures else []
    (out/'report_metrics.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path); parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args(); result=build_report(args.root,figures=not args.no_figures)
    print(json.dumps({'part1_validation':result['part1_validation'],'part2_validation':result['part2_validation'],
                      'reason_provenance':result['part2_reason_provenance']},indent=2))
