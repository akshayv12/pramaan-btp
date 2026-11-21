# Release verification

Prepared on 22 November 2026 using a fresh, isolated **Python 3.12.16 environment on macOS arm64**. The original working project's virtual environment was not reused for the runtime tests.

| Check | Result |
|---|---|
| Fresh `pip install -r requirements.txt` | Passed; tested versions recorded in `constraints-py312.txt` |
| `pip check` | No broken requirements |
| Task 5 standalone suite | 142 passed |
| Website/backend/HTTP suite | 48 passed |
| Integrated ranking, handoff and Tasks 1–5 suite | 277 passed, 13 skipped |
| JavaScript frontend suite | 12 passed |
| No-key constructed contract CLI | Generated JSON and text with citations/support explanation |
| Package build | Wheel includes all 24 Task 5 Python modules and `support_policy.json` |
| Installed wheel in a second empty environment | Source-grounded answer generated; packaged support policy loaded |
| Actual `website/server.py` launch | Frontend and HTTP demo passed on an unused loopback port; returned Hindi answer with two sources |
| Source preservation | 42 Tasks 1–4 runtime source/configuration files match the working project; Task 5 copies match |
| Publication scan | No known local API credentials, common token/private-key patterns, personal home paths or roll number found in release files |

The integrated tests include a real five-stage run against local HTML/PDF sources and a native-Tesseract mixed-PDF check. Only the external search-result list is substituted in the five-stage test; extraction, ranking and Task 5 run their real code.

The 13 skips cover seven optional BGE model tests, two absent historical Task 3 replay cases, and four legacy Task 4 tests gated on a separate `.venv` directory. Historical user runs are deliberately absent. Independent Linux CI is configured, but local verification does not imply that hosted CI has already passed. Windows interpreter layout is unit-tested; a complete Windows execution was not performed here.

Gemini generation, Qwen downloads/inference, a fresh NLI model download and real semantic BGE ranking were not repeated for this publication check. Tests mock external model replies and verify their contracts; they do not establish factual accuracy or per-language human calibration. Existing model-judged support-policy metadata is preserved without publishing private pilot logs.

Release-only changes cover setup/documentation, safe fixtures, packaging of the support policy, the Task 5 sync helper, Windows interpreter selection, a consistent website port and the no-model default. Tasks 1–4 runtime source/configuration and Task 5 generation/reliability algorithms were copied unchanged. See [architecture](ARCHITECTURE.md#release-preservation) for fixture changes.
