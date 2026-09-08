import sys
from pathlib import Path

# Make the `src` package importable from tests without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent))
