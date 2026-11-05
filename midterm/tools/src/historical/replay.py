"""Review or explicitly replay preserved historical experiments, without promotion.

Default CLI action checks source and input availability without fitting:
  python tools/src/historical/replay.py --check
Explicit training (may take substantial time):
  python tools/src/historical/replay.py --study label_shift --run --output models/new_experiments/label_shift_replay
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from threadpoolctl import threadpool_limits

from _historical_support import validate_output

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
STUDIES = {
    'graph': ('midterm_graph_refinement', 'run_geometry', 'models/part1/exploration/graph/summary.json'),
    'selective_pseudo': ('midterm_rank_part1', 'run', 'models/part1/exploration/selective_pseudo/summary.json'),
    'structure_geometry': ('midterm_part1_round2', 'run', 'models/part1/exploration/structure_geometry/summary.json'),
    'label_shift': ('midterm_part1_label_shift', 'run', 'models/part1/exploration/label_shift/backtest_decision.json'),
    'other_views': ('midterm_part1_other_views', 'run', 'models/part1/exploration/other_views/decision.json'),
    'mixture_followup': ('midterm_part1_mixture_followup', 'run', 'models/part1/exploration/structure_geometry/mixture_followup/status.json'),
    'val_refit_weight': ('midterm_part1_val_refit_weight', 'run', 'models/part1/exploration/structure_geometry/validation_refit_weight/summary.json'),
    'parity': ('midterm_part1_parity', 'run', 'models/part1/02_kmeans_lr/evidence/baseline_parity/summary.json'),
    'local_followup': ('midterm_part1_local_followup', 'run', 'models/part1/exploration/local_followup/summary.json'),
    'ridge_boundary': ('midterm_part1_ridge_boundary', 'run', 'models/part1/exploration/ridge_boundary/summary.json'),
    'local_selection': ('midterm_part1_local_consensus', 'main', 'models/part1/03_kernel_ridge/evidence/selection_procedure/summary.json'),
}
REQUIRED_INPUTS = (
    'data/part1/part1_train.csv', 'data/part1/part1_val.csv', 'data/part1/part1_test.csv',
    'data/part1/part1_sample_submission.csv', 'models/selection.json',
    'models/part1/01_supervised_lr/model.joblib',
    'models/part1/01_supervised_lr/evidence/original_run/validation_predictions.csv',
    'models/part1/02_kmeans_lr/model.joblib', 'models/part1/02_kmeans_lr/submission.csv',
    'models/part1/02_kmeans_lr/evidence/selected/selected_validation/validation_predictions.csv',
    'models/part1/03_kernel_ridge/model.joblib', 'models/part1/03_kernel_ridge/submission.csv',
    'models/part2/02_text_blend/submission.csv',
    'models/part1/exploration/local_followup/oof_predictions.csv',
    'models/part1/exploration/replay_inputs/oof_weight_char/fold_assignments.csv',
    'models/part1/exploration/replay_inputs/oof_weight_char/oof_weight_010.csv',
    'tools/archive/records/final_selection_nov01.json',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(root=ROOT):
    """Compile and import sources, and verify required saved inputs; never fit models."""
    root = Path(root).resolve()
    provenance = json.loads((DIRECTORY / 'source_provenance.json').read_text())
    for rec in provenance['source_files'].values():
        source = root / rec['ported_path']
        if digest(source) != rec['ported_sha256']:
            raise AssertionError(f'Historical source differs from migration record: {source.name}')
        compile(source.read_text(), str(source), 'exec')
    for rec in provenance['restored_evidence']:
        if digest(root / rec['path']) != rec['original_sha256']:
            raise AssertionError(f'Restored historical input differs: {rec["path"]}')
    for relative in REQUIRED_INPUTS:
        if not (root / relative).is_file():
            raise FileNotFoundError(f'Missing replay input: {relative}')
    for study, (module, function, summary) in STUDIES.items():
        target = importlib.import_module(module)
        if not callable(getattr(target, function, None)):
            raise TypeError(f'Missing replay function for {study}')
        if not (root / summary).is_file():
            raise FileNotFoundError(f'Missing recorded summary: {summary}')
    return {'status': 'PASS', 'ported_runners': len(provenance['source_files']),
            'replay_studies': len(STUDIES), 'training_performed': False,
            'check_scope': 'compile/import, source hashes, required saved inputs; no experimental refits'}


def recorded_summaries(root=ROOT):
    return {name: json.loads((Path(root) / item[2]).read_text()) for name, item in STUDIES.items()}


def run(study, root, output):
    """Explicit training API; keeps selected model/CSV/data bytes unchanged."""
    if study not in STUDIES:
        raise ValueError(f'Unknown historical study: {study}')
    root, output = validate_output(root, output)
    check(root)
    protected = {p: digest(p) for directory in ('data', 'models')
                 for p in (root / directory).rglob('*') if p.is_file()}
    protected[root / 'models/selection.json'] = digest(root / 'models/selection.json')
    module, function, _ = STUDIES[study]
    try:
        if study == 'local_selection':
            subprocess.run([sys.executable, str(root / 'tools/src/midterm_part1_local_consensus.py'),
                            '--root', str(root), '--out', str(output)], check=True, cwd=root)
        else:
            replay_function = getattr(importlib.import_module(module), function)
            if study in ('local_followup', 'ridge_boundary'):
                # Preserve these two original CLI wrappers' single-thread scope.
                with threadpool_limits(limits=1):
                    replay_function(root, output)
            else:
                replay_function(root, output)
    finally:
        changed = [str(p.relative_to(root)) for p, sha in protected.items()
                   if not p.is_file() or digest(p) != sha]
        if changed:
            raise RuntimeError(f'Curated inputs changed during historical replay: {changed}')
    return {'study': study, 'output': str(output), 'production_selection_changed': False}


def main(default_study=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--study', choices=tuple(STUDIES), default=default_study)
    parser.add_argument('--check', action='store_true', help='Check source and saved inputs only (default).')
    parser.add_argument('--run', action='store_true', help='Explicitly fit the selected historical experiment.')
    parser.add_argument('--output', type=Path, help='Required with --run; must not exist.')
    args = parser.parse_args()
    if args.run:
        if args.check or not args.study or args.output is None:
            parser.error('--run requires --study and --output, without --check.')
        result = run(args.study, args.root, args.output)
    else:
        result = check(args.root)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
