"""Portable entry point for the project MoralCLIP R@1 benchmark."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.evaluate_retrieval import main

if __name__ == '__main__':
    raise SystemExit(main())
