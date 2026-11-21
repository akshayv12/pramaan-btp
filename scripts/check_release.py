"""Check preserved upstream files and complete, identical Task 5 packages."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def errors(root=ROOT):
    problems = []
    manifest = json.loads((root / "docs/source_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append("Preserved upstream file differs: " + name)
    canonical = root / "task5_standalone/task5"
    mirror = root / "integrated_pipeline/task5"
    names = {p.name for p in canonical.glob("*.py")} | {"support_policy.json"}
    mirrored = {p.name for p in mirror.glob("*.py")} | {"support_policy.json"}
    if names != mirrored:
        problems.append("Task 5 module lists differ")
    for name in sorted(names):
        a, b = canonical / name, mirror / name
        if not a.is_file() or not b.is_file() or a.read_bytes() != b.read_bytes():
            problems.append("Task 5 mirror differs: " + name)
    return problems


def main():
    problems = errors()
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print("Preserved upstream source and Task 5 mirrors verified.")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
