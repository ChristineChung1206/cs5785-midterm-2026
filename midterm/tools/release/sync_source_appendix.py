"""Check or refresh the core/report source appendix in notebook 05.

The notebook retains executable study entries and also exposes the full source
for review without requiring a separate editor. Default mode only checks it.
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = 'tools/notebooks/05_historical_experiments.ipynb'
TAG = 'core_source_appendix'


def sources(root):
    return sorted([root / 'reproduce.py', * (root / 'tools/src').glob('*.py'),
                   * (root / 'report/scripts').glob('*.py')])


def cells(root):
    result = [{'cell_type': 'markdown', 'metadata': {TAG: True},
               'source': '## Core models and report source appendix\n\n'
               'Full Python source for the core experiments, saved-model reproduction, '
               'and figure/table generation follows. These review copies are checked '
               'against the executable files; model training remains opt-in.\n'}]
    result.append({'cell_type': 'code', 'metadata': {TAG: True},
                   'execution_count': None, 'outputs': [], 'source':
                   "import runpy\nsource_check = runpy.run_path(str(ROOT/'tools/release/sync_source_appendix.py'))\n"
                   "display(source_check['check'](ROOT))\n"})
    for path in sources(root):
        relative = str(path.relative_to(root))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result.append({'cell_type': 'markdown',
                       'metadata': {TAG: True, 'source_path': relative, 'sha256': digest},
                       'source': f'### {relative}\n\nSHA-256: `{digest}`\n\n```python\n'
                       + path.read_text() + '\n```\n'})
    return result


def check(root=ROOT):
    root = Path(root).resolve()
    notebook = json.loads((root / NOTEBOOK).read_text())
    recorded = [cell for cell in notebook['cells'] if cell['metadata'].get(TAG)]
    expected = cells(root)
    if len(recorded) != len(expected):
        raise AssertionError('Source appendix is incomplete; run this tool with --write.')
    for actual, wanted in zip(recorded, expected):
        content = actual['source']
        content = content if isinstance(content, str) else ''.join(content)
        if content != wanted['source'] or actual['metadata'] != wanted['metadata']:
            raise AssertionError('Source appendix is stale; run this tool with --write.')
    return {'status': 'PASS', 'core_and_report_source_files': len(sources(root)),
            'historical_source': 'Included separately above, with migration hashes.',
            'training_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.write:
        path = ROOT / NOTEBOOK
        notebook = json.loads(path.read_text())
        notebook['cells'] = [cell for cell in notebook['cells']
                             if not cell['metadata'].get(TAG)] + cells(ROOT)
        path.write_text(json.dumps(notebook, indent=1) + '\n')
    print(json.dumps(check(), indent=2))


if __name__ == '__main__':
    main()
