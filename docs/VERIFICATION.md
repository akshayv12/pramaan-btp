# Layout revision verification

Checked on 22 November 2026 with Python 3.12.16 on macOS arm64. The baseline dependency set is unchanged. The editable package was reinstalled after relocation, and a separate wheel installation was checked outside the repository.

| Check | Result |
|---|---|
| Shared Task 5 suite | 142 passed |
| Website/backend/HTTP/configuration suite | 49 passed |
| Task 1/2/4 and cross-task integration suite | 263 passed, 9 skipped |
| JavaScript frontend | 12 passed |
| Total executed checks | 466 passed |
| `pip check` | No broken requirements |
| Source preservation | 62 unchanged files; four explicit task/path-configuration edits recorded |
| Duplicate audit | 27 byte-identical pairs verified; 29 redundant/obsolete files removed |
| Public resume CLI | Saved Task 4 evidence generated cited JSON/text through the shared package |
| Actual website launch | Frontend and HTTP Hindi example passed using the relocated pipeline |
| Wheel | One Task 5 package, all 24 modules and support policy; no mirror or test packages |

The integrated suite runs the real five-stage pipeline against local HTML/PDF sources; only the external search-result list is substituted. Native Tesseract mixed-PDF extraction also passed. The four lexical stage tests formerly gated on a separate Task 4 environment now execute using the installed root environment.

Nine skips cover seven optional BGE model tests and two historical Task 3 replay cases whose private inputs are not published. The lower count compared with the initial 479 checks reflects removal of 18 duplicated handoff checks, execution of four formerly skipped checks and one added shared-environment regression check; unique coverage was retained.

The initial publication's GitHub Linux CI passed. The reorganised commit has the same offline workflow with updated paths; its hosted result is available in the repository's Actions tab. Local verification does not itself establish a hosted CI result. Full Windows execution, fresh Gemini/Qwen inference, NLI weight download and real BGE ranking were not repeated for this folder change. Existing model-review/calibration limitations remain unchanged.

The original working project, report and private model/history folders outside this Git checkout were not edited. See [cleanup audit](CLEANUP_AUDIT.md) and [source manifest](source_manifest.json) for the exact preservation scope.
