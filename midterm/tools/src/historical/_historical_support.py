"""Filesystem/import adapters for preserved historical experiments; no model code."""
from pathlib import Path
import sys

CORE = Path(__file__).resolve().parent.parent
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


def validate_output(root, output):
    """Allow only a new output directory, outside curated data/model/report/tools trees."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if not (root / 'models/selection.json').is_file():
        raise FileNotFoundError('Expected a complete midterm project with models/selection.json.')
    if output.exists():
        raise FileExistsError(f'Historical replay requires a new, nonexistent directory: {output}')
    allowed = root / 'models/new_experiments'
    for name in ('data', 'models', 'report', 'tools'):
        if output.is_relative_to(root / name) and not output.is_relative_to(allowed):
            raise ValueError('Do not write into curated evidence. Use models/new_experiments/NEW_NAME or a new external directory.')
    return root, output


def prepare_output(root, output):
    root, output = validate_output(root, output)
    output.mkdir(parents=True, exist_ok=False)
    return root, output
