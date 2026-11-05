"""Verify all files in an extracted midterm_code.zip against its included manifest."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def verify(root):
    root = root.resolve()
    manifest = root / 'tools/release/package_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError('This check is for the extracted code ZIP; no package manifest exists here.')
    records = json.loads(manifest.read_text())['files']
    expected = set(records) | {'tools/release/package_manifest.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'}}
    if actual != expected:
        raise AssertionError(f'Package inventory mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    for relative, record in records.items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise AssertionError(f'Missing or invalid packaged file: {relative}')
        if p.stat().st_size != record['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != record['sha256']:
            raise AssertionError(f'Packaged bytes changed: {relative}')
    return {'status': 'PASS', 'verified_files': len(records)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.project_root), indent=2))
