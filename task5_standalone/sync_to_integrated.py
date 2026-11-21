"""Replace only the owned Task 5 mirror and its handoff tests; never edit Tasks 1-4."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
TARGET=ROOT.parent/'integrated_pipeline'


def main():
    if not (TARGET/'task4_rank_evidence.py').is_file():raise SystemExit('Integrated pipeline not found; nothing changed')
    source=ROOT/'task5';destination=TARGET/'task5'
    incoming={p.name for p in source.glob('*.py')} | {'support_policy.json'}
    if not {'pipeline.py','handoff.py','__init__.py'}.issubset(incoming):raise SystemExit('Incomplete canonical package; nothing changed')
    if not (source/'support_policy.json').is_file():raise SystemExit('Missing support policy; nothing changed')
    destination.mkdir(exist_ok=True)
    for old in destination.iterdir():
        if old.name not in incoming:
            if old.is_dir():shutil.rmtree(old)
            else:old.unlink()
    for name in incoming:shutil.copy2(source/name,destination/name)
    tests=TARGET/'integration_tests';tests.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'tests/test_handoff.py',tests/'test_task5_handoff.py')
    print('Replaced Task 5 mirror and Task 5 handoff tests only. Tasks 1-4 untouched.')


if __name__=='__main__':main()
