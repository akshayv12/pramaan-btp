from task5.handoff import main
from task5.config import load_env_file
from pathlib import Path

if __name__ == "__main__":
    load_env_file(Path(__file__).resolve().parent / ".env")
    raise SystemExit(main())
