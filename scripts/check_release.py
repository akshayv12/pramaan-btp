"""Check preserved task code, recorded path edits and the single Task 5 package."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def errors(root=ROOT):
    problems = []
    manifest = json.loads((root / "docs/source_manifest.json").read_text(encoding="utf-8"))
    expected_files = {**manifest["files"], **{name: item["sha256"] for name, item in manifest["layout_changes"].items()}}
    for name, expected in expected_files.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append("Recorded task source differs: " + name)
    for name in ("__init__.py", "__main__.py", "pipeline.py", "handoff.py", "support_policy.json"):
        if not (root / "task5" / name).is_file():
            problems.append("Incomplete shared Task 5 package: " + name)
    for legacy in ("integrated_pipeline/task5/pipeline.py", "task5_standalone/task5/pipeline.py"):
        if (root / legacy).is_file():
            problems.append("Redundant Task 5 implementation remains: " + legacy)
    return problems


def main():
    problems = errors()
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print("Preserved task source, recorded path edits and single Task 5 package verified.")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
