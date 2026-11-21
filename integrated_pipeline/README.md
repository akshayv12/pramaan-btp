# Integrated Tasks 1–5

Use the repository's [README](../README.md) for installation and browser/CLI examples. Run commands from the repository root.

`run_integrated_pipeline.py` accepts one of:

- `--query "YOUR QUESTION"`: run all stages.
- `--task3-input FILE.json`: run Task 4 and Task 5 on extracted evidence.
- `--task4-input FILE.json`: run Task 5 on already ranked evidence.

Use `--lexical-only` for the existing Task 4 TF-IDF baseline without BGE-M3. Omit it only after preparing the optional semantic model. Task 4 options are rejected when using `--task4-input`, because that evidence has already been ranked.

The source files, architecture and production configuration of Tasks 1–4 were copied unchanged from the working project for this release. [The manifest](../docs/source_manifest.json) records their SHA-256 checksums. Fixture paths were made portable; private extraction outputs, reports and historical experiments were excluded. The Task 5 mirror is checked against `task5_standalone/task5/`.

The unified runner preserves the upstream subprocess boundaries, isolates each live run's handoffs, and records the execution mode. See [the contract guide](../docs/ARCHITECTURE.md) for more detail.
