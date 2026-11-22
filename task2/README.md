# Task 2 — Web search and document collection

**Input:** Task 1's question metadata and search-query variants.

**Work:** use the existing Brave/DuckDuckGo/Bing cascade, collect eligible URLs, download supported documents, merge duplicates and record source/access information. Search-result snippets do not replace downloaded evidence.

**Output:** `task3_input.json`: document manifest, saved paths, URLs/source metadata and processing/OCR requirements for Task 3.

| File | Role |
|---|---|
| `main.py` | Search/collection, downloads, access decisions and manifest handling |
| `search_engines.py` | Existing provider adapters and fallbacks |
| `tests/test_task2_integration.py` | Query metadata, manifest and provider checks |

The collector and search algorithms are unchanged. Cross-task orchestration now lives in `pipeline/`.

```sh
python -m pytest task2/tests -q
python pipeline/run_pipeline.py --query "Tell me about Sundara Kanda" --max-queries 4 --top-k 3
```

The second command runs Tasks 1–2, makes live requests and writes handoffs/downloads in the current directory. Prefer the [public pipeline CLI](../pipeline/README.md) for isolated end-to-end runs and shared `.env` loading.

**Presentation line:** “Task 2 finds and collects documents, preserving where each document came from.” Next: [Task 3](../task3/README.md).
