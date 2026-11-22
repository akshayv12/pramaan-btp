# Pipeline — Connect Tasks 1–5

This folder connects the separate tasks while retaining their algorithms and subprocess boundaries. Each public live run has its own document/handoff directory. The website, public runner and Task 5 use the shared repository-root `.env`.

## Public entry point

From the repository root in the installed environment:

```sh
python pipeline/run_integrated_pipeline.py --query "Tell me about Sundara Kanda" --lexical-only --provider extractive --output-dir outputs/demo
```

Choose one input:

- `--query "QUESTION"`: run Tasks 1–5.
- `--task3-input FILE.json`: resume with Task 4 and Task 5.
- `--task4-input FILE.json`: generate from evidence already ranked by Task 4.

Use `--provider gemini --strict` or `--provider ollama --strict` after model setup, and `--language hi` for Hindi. Use `--lexical-only` when BGE-M3 is not prepared. Task 4 settings are rejected when resuming after Task 4.

## Why two runners remain

`run_pipeline.py` implements the existing stage handoffs and supports stopping after Task 2/3/4 or calling Task 5. `run_integrated_pipeline.py` is the public live/resume interface: it adds output isolation, effective configuration, resume modes and final execution metadata while calling the lower-level runner. They are distinct orchestration layers, not duplicate task implementations.

`tests/` exercises partial/fatal failures, overwrite protection, local HTML/PDF collection and the real five-stage handoff. Task-specific tests stay in the task folders. The byte-identical extra Task 5 handoff test was removed; its coverage remains in `task5/tests/test_handoff.py`.

```sh
python -m pytest task1/tests task2/tests task4/tests pipeline/tests -q
```

**Presentation line:** “Pipeline integration connects the tasks' input/output contracts, tracks failures and delivers one final result to the website.”
