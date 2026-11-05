"""Build the four current deliverables into a new directory; never train models.

Run from any working directory:
    python tools/release/build_release.py --output-dir /path/to/new-deliverables
The destination must not already exist. Historical full ZIP snapshots, caches,
and local release receipts are excluded from midterm_code.zip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = 'tools/release/current_release.json'
PACKAGE_MANIFEST = 'tools/release/package_manifest.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def included(relative):
    path = Path(relative)
    parts = path.parts
    if any(x in {'__pycache__', '.ipynb_checkpoints', '.git'} for x in parts):
        return False
    if path.name == '.DS_Store' or path.suffix in {'.pyc', '.pyo'}:
        return False
    if str(path).startswith('report/build/'):
        return False
    if path.parent == Path('report') and path.suffix in {'.aux', '.log', '.out', '.gz'}:
        return False
    if str(path).startswith('tools/archive/') and path.suffix == '.zip':
        return False
    return str(path) not in {RECEIPT, PACKAGE_MANIFEST}


def verify_report(root):
    directory = root / 'report'
    layout = json.loads((directory / 'verification/layout_verification.json').read_text())
    pdf = directory / 'final_report.pdf'
    if layout.get('status') != 'PASS' or digest(pdf) != layout['pdf_sha256']:
        raise AssertionError('Report PDF does not match its verified artifact.')
    if layout['body_pages'] > 5 or layout.get('total_pages') != 6:
        raise AssertionError('Unexpected report page count.')
    for relative, expected in {**layout['source_sha256'], **layout.get('figure_sha256', {})}.items():
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory.resolve()):
            raise ValueError('Report source must remain inside report/.')
        if digest(candidate) != expected:
            raise AssertionError(f'Report source changed after layout verification: {relative}')
    provenance_path = directory / 'verification/source_provenance.json'
    if digest(provenance_path) != layout['source_provenance_sha256']:
        raise AssertionError('Report provenance changed after layout verification.')
    if digest(directory / 'verification/compile.log') != layout['compile_log_sha256']:
        raise AssertionError('Report compile log changed after verification.')
    provenance = json.loads(provenance_path.read_text())
    for key in ('current_selection', 'report_metrics'):
        rec = provenance[key]
        if digest(root / rec['path']) != rec['sha256']:
            raise AssertionError(f'Report evidence changed after verification: {key}')
    return pdf


def build(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing destination: {output}')
    if output.is_relative_to(root):
        raise ValueError('Deliverables must be outside midterm/ to avoid packaging recursion.')
    registry = json.loads((root / 'models/selection.json').read_text())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    run = subprocess.run([sys.executable, str(root / 'reproduce.py'), '--project-root', str(root)],
                         cwd=root, env=env, text=True, capture_output=True)
    if run.returncode:
        raise RuntimeError('Saved-model verification failed:\n' + run.stdout + run.stderr)
    verification = json.loads(run.stdout)
    if verification.get('status') != 'PASS':
        raise AssertionError('Saved-model verification did not report PASS.')
    source_checks = {}
    for name, relative in [('historical_sources', 'tools/src/historical/replay.py'),
                           ('notebook_source_appendix', 'tools/release/sync_source_appendix.py')]:
        check = subprocess.run([sys.executable, str(root / relative)], cwd=root,
                               env=env, text=True, capture_output=True)
        if check.returncode:
            raise RuntimeError(f'{name} verification failed:\n' + check.stdout + check.stderr)
        source_checks[name] = json.loads(check.stdout)
    pdf = verify_report(root)
    files = sorted(p for p in root.rglob('*') if p.is_file() and included(p.relative_to(root)))
    manifest = {'format_version': 1, 'purpose': 'Verify extracted active project before running commands that regenerate outputs.',
                'excludes': ['full historical ZIP snapshots', 'Python/Notebook caches', 'report build intermediates',
                             'current_release.json', 'the manifest itself'],
                'files': {str(p.relative_to(root)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in files}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.midterm-release-', dir=output.parent) as temp:
        staging = Path(temp) / 'deliverables'
        staging.mkdir()
        for part in ('part1', 'part2'):
            record = registry[part]['best_public']
            source = root / record['submission']
            if digest(source) != record['sha256']:
                raise AssertionError(f'Canonical {part} submission bytes changed.')
            shutil.copy2(source, staging / (part + '.csv'))
        shutil.copy2(pdf, staging / 'final_report.pdf')
        with zipfile.ZipFile(staging / 'midterm_code.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in files:
                z.write(p, 'midterm/' + str(p.relative_to(root)))
            z.writestr('midterm/' + PACKAGE_MANIFEST, json.dumps(manifest, indent=2) + '\n')
        # Verify every archive member against the source inventory before publication.
        with zipfile.ZipFile(staging / 'midterm_code.zip') as z:
            expected = {'midterm/' + rel for rel in manifest['files']} | {'midterm/' + PACKAGE_MANIFEST}
            if set(z.namelist()) != expected:
                raise AssertionError('Unexpected package member set.')
            for rel, rec in manifest['files'].items():
                if hashlib.sha256(z.read('midterm/' + rel)).hexdigest() != rec['sha256']:
                    raise AssertionError(f'Archive verification failed: {rel}')
        receipt = {'status': 'PASS', 'training_performed': False, 'uploads_performed': False,
                   'public_scores': {p: registry[p]['best_public']['public_score'] for p in ('part1', 'part2')},
                   'submission_identity': 'Exact bytes copied from checksum-verified best-public artifacts.',
                   'saved_model_verification': verification, 'source_checks': source_checks, 'package_files': len(manifest['files']) + 1,
                   'full_historical_snapshots_included': False,
                   'files': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in staging.iterdir()}}
        staging.rename(output)
    (root / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
