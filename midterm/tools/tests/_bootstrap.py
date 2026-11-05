"""Make curated src modules importable under unittest discovery or direct tests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / 'tools/src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
