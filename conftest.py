"""Resolve the existing stage modules for repository-wide integration tests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for folder in ("task1", "task2", "task4", "pipeline"):
    sys.path.insert(0, str(ROOT / folder))
