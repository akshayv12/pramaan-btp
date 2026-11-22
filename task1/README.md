# Task 1 — Multilingual query generation

**Input:** the user's question and a limit on search-query variants.

**Work:** detect Hindi, Roman Hindi, English or mixed input; normalise text; identify entities/key terms and constraints; generate bounded search variants while preserving the original intent. Existing baseline rules and optional backends are retained.

**Output:** question metadata and query variants, saved by the pipeline as `task1_input.json` for Task 2.

| File | Role |
|---|---|
| `app.py` | Query-generation functions, data models and FastAPI endpoints |
| `streamlit_app.py` | Optional isolated Task 1 demo; separate from the main website |
| `tests/test_app.py` | Language, intent and output-contract checks |
| `requirements.txt` | Optional API/demo dependencies |

From the repository root after the standard installation:

```sh
python -c "from task1.app import generate_queries; print(generate_queries('Tell me about Sundara Kanda', 4).model_dump_json(indent=2))"
python -m pytest task1/tests -q
```

For the optional API, install this folder's requirements and run `python -m uvicorn app:app --app-dir task1 --port 8001`. The isolated UI uses `python -m streamlit run task1/streamlit_app.py`. Neither is required by Pramaan's website.

**Presentation line:** “Task 1 converts the original question into search variants; it does not generate the final answer.” Next: [Task 2](../task2/README.md).
