# Local setup and troubleshooting

Use [the main quick start](../README.md#quick-start) first. Commands below run from the repository root, using the installed Python 3.12 environment.

## Before a live search

The no-key saved example checks the frontend, backend and Task 4 → Task 5 handoff without contacting a search service. A live question additionally needs the integration dependencies and internet access. For natural-language synthesis, configure Gemini or download the local Ollama model. OCR and independent NLI are separate optional installations.

Check the environment:

```sh
python --version
python -m pip check
python -c "import task5, numpy, sklearn, requests, fastapi; print('Pipeline imports OK')"
python scripts/check_release.py
```

The browser's backend status and `GET /api/health` report which components are ready. The server must run in the same environment as the installed package. `PIPELINE_PYTHON` can explicitly select another pipeline interpreter.

## API configuration

The preferred Gemini setup uses a hidden prompt:

```sh
python website/configure_gemini.py --model YOUR_AVAILABLE_MODEL_ID
```

Alternatively, copy `.env.example` to `integrated_pipeline/.env` and edit it locally. The website and integrated runner load that file; the standalone entry point also loads `task5_standalone/.env`. Existing process environment variables take priority. Restart the backend after editing persistent configuration.

Relevant settings:

| Variable | Meaning |
|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Your backend-only Gemini credential |
| `GEMINI_MODEL` | Model available to your account; pass explicitly during setup |
| `TASK5_PROVIDER` | `auto`, `gemini`, `ollama`, or `extractive` |
| `TASK5_VERIFICATION` | `basic`, `standard`, or `thorough` |
| `TASK5_JUDGE_PROVIDER` | `same`, `gemini`, or `ollama` for the CLI review provider |
| `TASK5_JUDGE_MODEL` | Optional model override for review |
| `BRAVE_API_KEY` | Optional key for the existing Task 2 search cascade |
| `PIPELINE_PYTHON` | Interpreter used by the website's pipeline subprocesses |
| `TASK5_NLI_DIRECTORY` | Override the local verifier model directory |
| `TASK5_CALIBRATION_PATH` | Optional file produced by the human-review calibration tool |

The website offers its own per-search provider/review settings. When Gemini is ready, the website defaults to Gemini review, including for a Qwen draft; select same-provider review to keep synthesis/review local. Standard mode normally makes selection, drafting and review calls. Thorough adds a sampled alternative. API charges and quotas belong to your account.

## Semantic ranking with BGE-M3

This optional setup retains the existing Task 4 implementation and production profile. The default lightweight installation supports lexical ranking and does not include Torch or BGE-M3 weights.

1. Install a suitable PyTorch build for your machine using the [official installer](https://pytorch.org/get-started/locally/). CPU execution is supported by the project's profile.
2. Install the existing semantic dependencies:

   ```sh
   python -m pip install -r integrated_pipeline/requirements.txt
   python integrated_pipeline/scripts/prepare_model.py --config integrated_pipeline/configs/cpu.json
   ```

3. Restart the website and choose semantic ranking after readiness is reported.

The helper downloads the pinned model to `integrated_pipeline/.venv/models/bge-m3/`, checks an embedding and writes a local manifest. Its `.venv` model-storage folder is separate from the root `.integration-venv` interpreter. Expect a multi-GB download. The helper may normalise the selected model configuration when explicitly run; the distributed production ranking weights are unchanged. Real semantic-model tests are opt-in, using `--run-model-tests` with the integrated pytest suite.

## Common problems

| Symptom | Check / action |
|---|---|
| `No module named task5` | Run `python -m pip install -e .` from the repository root with the interpreter that runs the server. |
| Live search is unavailable | Install `requirements.txt` in the pipeline environment; check `/api/health`. |
| Local model unavailable | Check `ollama --version`, run `python website/setup_llm.py`, and restart. If another service owns port 11434, its model store is used. |
| Gemini is not ready | Check model access, key, API quota and network; run `python website/setup_gemini.py`. Listing a model does not guarantee generation quota. |
| NLI unavailable | Install `requirements-verification.txt`, run `website/setup_verification.py`, restart; missing verification remains visible. |
| Image/scanned sources fail | Check `tesseract --version`, run `python website/setup_ocr.py`, restart. Transformer OCR engines are optional and not installed by the basic setup. |
| Answer is in the evidence language | Source extraction does not translate. Select an LLM provider for a translated answer. |
| Answer has Low support | Read the support explanation, source coverage and verification details. A support label is not determined only by the draft's fluency. |
| Search is slow / times out | Web collection and OCR add latency. Use a saved Task 4 handoff to inspect synthesis separately; Basic verification reduces verification work. |
| Port is occupied | Start with `--port 8791` and open `http://127.0.0.1:8791`. |

The website limits concurrent work and each job's runtime. Inspect the stage trace and local per-job log to see whether collection, extraction or answer generation failed. Do not upload private run logs or `.env` files to public issues.

## Evaluation tools

Task 5 includes bounded experiments for passage count/order and direct cross-lingual versus translate-first synthesis:

```sh
python -m task5.experiments task5_standalone/examples/contract_example.json --output-dir outputs/experiment
```

The default creates a plan only. `--execute` explicitly enables model calls. Use saved evidence to compare generation settings without rerunning retrieval. `python -m task5.human_review --help` exposes blinded review export and calibration. Human review starts pending; the tool does not manufacture human labels.

## Clean local data

Stop the website before manually removing generated history or weights. `website/.runs/` contains searches and collected evidence; `.models/` contains local weights; `.ocr/` contains OCR data; `.services/` contains managed-service state. These folders and `.env` files are ignored by Git. Removing weights requires repeating the relevant setup download.
