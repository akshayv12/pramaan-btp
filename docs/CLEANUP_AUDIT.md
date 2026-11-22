# Repository layout and cleanup audit

The repository was checked before changing the initial published commit, `2a85c2c`. Files were compared byte-for-byte using SHA-256, their imports/CLI references were inspected, and the task/pipeline tests were rerun after relocation.

## Verified removals

| Removed | Files | Why removal is safe |
|---|---:|---|
| Integrated Task 5 mirror | 25 | All 24 Python modules and the support-policy JSON matched the canonical copy exactly. The website and pipeline now import one `task5/` package. |
| Extra Task 5 handoff test | 1 | Byte-identical to the retained `task5/tests/test_handoff.py`; it ran the same 18 checks twice. |
| Repeated Task 5 CLI wrappers | 2 | The wrappers were identical and forwarded to `task5.handoff.main`. The existing `python -m task5` and installed `btp-task5` entry points provide that same CLI. The stage runner now uses the module entry point. |
| Task 5 mirror-sync script | 1 | Its only role was maintaining the removed duplicate package/test copy. There is no second implementation to sync. |
| **Total** | **29** | No processing capability was removed. |

[cleanup_audit.json](cleanup_audit.json) records the 27 verified identical pairs, their hashes and replacement/retained paths. Both wrappers were removed after comparison; their replacement is a module entry point rather than another copied file.

## Relocation and preservation

The layout is `task1/`, `task2/`, `task3/`, `task4/`, `task5/`, `pipeline/` and `website/`. Cross-task orchestration/tests moved out of the Task 2 folder. Each task now has its own input/process/output README.

The preservation manifest records **62 unchanged processing/configuration files** containing **343 unchanged function definitions**. Four recorded files need layout/configuration edits: the two pipeline runners, Task 5's verifier-directory constant and its CLI environment loading. Query-generation, collection, extraction, ranking, diversity selection, generation and reliability algorithms remain the same. Website/test/package paths and documentation were updated to consume the new structure.

API settings are written once to the ignored repository-root `.env`, rather than duplicated across integrated/standalone folders. A configuration regression test checks that existing settings are preserved, the key is not printed and the file is protected on POSIX systems.

## Deliberately retained

- **Two pipeline runners:** the lower-level stage handoff runner and the public live/resume/output-isolation CLI perform different jobs. Their roles are explained in `pipeline/README.md`.
- **Two lexical profiles:** the integration profile retains its relevance gate; the website profile disables it for Task 5 inspection. Their filtering settings differ, so merging them would change behaviour. Both now live under `task4/configs/`.
- **Optional OCR/model and evaluation helpers:** infrequent execution is not evidence of redundancy. They implement different routes or user-supplied-data experiments and are kept with their task.
- **Regression evidence:** the fixtures cover different observed failures; they are not interchangeable duplicates. Original quotes, scores and IDs are preserved.
- **Task 1's isolated demo:** it exposes Task 1 alone and is distinct from the complete Pramaan website; it remains optional.

The obsolete requirement that a separate Task 4 `.venv` directory exist before running four lexical integration tests was removed. Those tests use the root installed environment and no BGE model, and now execute normally. Optional real-BGE tests and historical-data cases remain explicit skips when their inputs are absent.

## Commands after this change

| Before | Now |
|---|---|
| `python integrated_pipeline/run_integrated_pipeline.py ...` | `python pipeline/run_integrated_pipeline.py ...` |
| `python task5_standalone/process_task4_output.py ...` | `python -m task5 ...` |
| Integrated/standalone `.env` copies | One repository-root `.env` |
| Edit Task 5 and run a mirror sync | Edit `task5/` once |

Reports, credentials, private history, downloads and model weights remain excluded from Git. Runtime files and the original working project outside this Git checkout were not deleted. See [verification](VERIFICATION.md) for the checks actually performed.
