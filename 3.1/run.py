"""Run section 3.1 from any working directory with Python 3.11+."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moral_llava.pipeline import main

if __name__ == '__main__':
    raise SystemExit(main())
