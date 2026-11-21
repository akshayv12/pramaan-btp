# Pramaan website

Start with the repository's [local setup guide](../README.md#quick-start).

From the repository root, using the installed Python environment:

```sh
python website/server.py --port 8790
```

Open [http://127.0.0.1:8790](http://127.0.0.1:8790). The frontend is plain HTML/CSS/JavaScript and the backend uses Python's local HTTP server. No npm build or database is needed.

- A fresh query runs the existing Tasks 1–5 subprocess pipeline.
- **Use Task 4 evidence** accepts a single-query or batch JSON handoff and bypasses Tasks 1–4.
- **Source extraction** works without a model or key and keeps the evidence language.
- **Gemini** and **Local Qwen3 4B** generate phrased answers with claim citations.
- The answer panel shows support reasoning, sources, verification details and downloads.

Runtime folders `.runs`, `.services`, `.models` and `.ocr` are generated locally and ignored by Git. API keys are read by the backend and are not sent to the browser as configuration responses. Gemini synthesis/review sends selected evidence to Google's API. The server listens on loopback only.

The pipeline interpreter is selected by `PIPELINE_PYTHON`, then the project's `.integration-venv` (including Windows `Scripts/python.exe`), then the server interpreter. You can override it using `--python /path/to/python`. `--runs-dir /path/to/folder` changes the local history location.

See [architecture and API](../docs/ARCHITECTURE.md) and [troubleshooting](../docs/LOCAL_SETUP.md).
