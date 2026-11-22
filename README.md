# Pramaan

**Ask a question, get a concise answer with sources, and see how strongly the evidence supports it.**

Pramaan is a BTP project for Indian-context multilingual web retrieval and summarisation. It includes the integrated Tasks 1–5 pipeline and a local browser website with a Python backend. Hindi/Devanagari and English are the initial answer languages; queries can also contain Roman Hindi.

This repository contains the runnable project, tests and example handoffs. Personal reports, presentations, credentials, downloaded documents, search history and model weights are excluded.

## What is included

- A browser frontend with a query box, language selection, search options and recent local searches.
- The existing query generation, collection, extraction/OCR and evidence ranking stages (Tasks 1–4).
- Task 5: diverse evidence selection, Gemini or local Qwen synthesis, claim citations and an explained High / Medium / Low support assessment.
- A Task 4 JSON upload mode, so you can generate an answer from evidence you already have.
- Explicit Source extraction mode for trying the pipeline without an LLM or API key.

```text
Question → Task 1: query generation → Task 2: web collection
         → Task 3: extraction / OCR → Task 4: ranked evidence
         → Task 5: select diverse evidence → draft → verify / review
         → Answer + supporting sources + explained support label
```

Task 4 owns retrieval ranking. Task 5 selects a bounded, diverse context from that ranked output; uploading Task 4 output bypasses Tasks 1–4. Each task has its own folder and README. Use [the presentation guide](docs/PRESENTATION_GUIDE.md) to explain the tasks in order, and [the cleanup audit](docs/CLEANUP_AUDIT.md) to review the verified removals.

## Quick start

Use **Python 3.12** for the full pipeline. Git is needed to clone the project. Node.js is only needed for the frontend tests; there is no npm build or database setup. `requirements.txt` uses the tested baseline versions in `constraints-py312.txt`; optional model installations are separate.

```sh
git clone https://github.com/akshayv12/pramaan-btp.git
cd pramaan-btp
```

### macOS / Linux

```sh
python3.12 -m venv .integration-venv
source .integration-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python website/server.py --port 8790
```

### Windows PowerShell

```powershell
py -3.12 -m venv .integration-venv
.\.integration-venv\Scripts\python.exe -m pip install --upgrade pip
.\.integration-venv\Scripts\python.exe -m pip install -r requirements.txt
.\.integration-venv\Scripts\python.exe website/server.py --port 8790
```

Open **[http://127.0.0.1:8790](http://127.0.0.1:8790)**. Keep the terminal open; press **Ctrl+C** to stop. On Windows, use `.\.integration-venv\Scripts\python.exe` in place of `python` in the commands below if the environment is not activated.

1. For a first check, open **Search options**, select **Source extraction**, and run the saved example. It needs no API key, model download or live search.
2. To phrase or translate an answer, configure **Gemini** or **local Qwen** below, then select that answer provider.
3. Type your question and search. The default available Task 4 method is the **lexical baseline** until the optional BGE-M3 model is prepared. A fresh search needs internet access.

Source extraction keeps the source language; it does not translate. A generated Hindi answer needs a configured LLM. An explicit instruction such as `tell me 5 points on Ramayana in Hindi` takes priority over a previous language-selector choice.

On macOS/Linux, you can alternatively manage the backend in the background from an activated environment:

```sh
python website/launch.py start
python website/launch.py status
python website/launch.py stop
```

The foreground command above is the recommended portable entry point. If port 8790 is busy, use `--port 8791` and open that address.

## Choose an answer model

### Gemini

Create your own Gemini API key and choose a model available to your account. Check the [official model list](https://ai.google.dev/gemini-api/docs/models); access and quota can differ between accounts.

```sh
python website/configure_gemini.py --model YOUR_AVAILABLE_MODEL_ID
python website/setup_gemini.py
```

The first command asks for the key using a hidden prompt, validates model access, and writes the shared, ignored **repository-root `.env`** file. The website, pipeline and Task 5 use this same configuration. Restart the website after persistent configuration changes. Never put a key in a Git commit, browser URL or command argument. The UI also offers backend-session configuration under **Search options → Configure Gemini**.

With Gemini selected, evidence is sent to Google's API for synthesis/review. Automatic selection prefers configured Gemini, then an available local model. A failed explicit model request is shown as an error rather than presented as a successful generated answer.

### Local Qwen with Ollama

Install [Ollama](https://docs.ollama.com/quickstart), ensure `ollama` is on your PATH, then run:

```sh
python website/setup_llm.py
```

This explicitly downloads `qwen3:4b-instruct-2507-q4_K_M`. Downloads are several GB and local inference needs sufficient RAM. The settings are in [website/configs/local_llm.json](website/configs/local_llm.json); downloaded weights stay in an ignored local directory. Searches never initiate a model download.

If an existing Ollama service is running, setup uses it; that service controls its own model directory. Otherwise the project starts an owned service using `website/.models/ollama/`. Restart the website after setup.

Select **Local Qwen3 4B** to draft locally. When Gemini is configured, the website can use **Gemini review** for a Qwen draft. Choose **same-provider review** for fully local answer generation and review. Live web retrieval still contacts search engines/source websites.

## Optional OCR and verification

| Component | Needed for | Setup |
|---|---|---|
| Tesseract + English/Hindi data | Scanned PDFs and images | Install Tesseract, then `python website/setup_ocr.py` |
| Multilingual ONNX NLI | Independent source-to-claim checking in Standard/Thorough mode | Commands below |
| BGE-M3 | Task 4 semantic ranking | [Advanced setup](docs/LOCAL_SETUP.md#semantic-ranking-with-bge-m3) |

Tesseract has a separate system executable and language data; see its [installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html). On macOS, `brew install tesseract` installs the runtime. On Ubuntu/Debian, use `sudo apt install tesseract-ocr`. On Windows, install a supported runtime and add its executable folder to PATH. The project setup command downloads checksum-pinned English/Hindi data.

```sh
python -m pip install -r task5/requirements-verification.txt
python website/setup_verification.py
```

The NLI setup downloads pinned quantised weights (about 339 MB) plus a tokenizer. Missing or failed verification is reported. Basic mode omits independent NLI; Thorough adds an alternative draft and a consistency signal. For a quick no-model demonstration, choose Source extraction.

## Run without the website

All commands run from the repository root in the installed environment.

```sh
# No-key, no-network contract demonstration using constructed evidence
python -m task5 \
  task5/examples/contract_example.json \
  --provider extractive --output outputs/demo.json --text-output outputs/demo.txt

# Use your own ranked Task 4 evidence; do not rerun retrieval
python pipeline/run_integrated_pipeline.py \
  --task4-input PATH_TO_TASK4_OUTPUT.json --provider gemini --strict \
  --language hi --output-dir outputs/from_task4

# Run Tasks 1–5 with the existing Task 4 lexical baseline
python pipeline/run_integrated_pipeline.py \
  --query "Tell me about Sundara Kanda" --lexical-only \
  --provider gemini --strict --task4-top-k 24 --output-dir outputs/live
```

For a live run without an LLM key, replace `--provider gemini --strict` with `--provider extractive`. Search results can be unavailable, weak or blocked; the system may return limited evidence. The CLI exits with 1 for a partial upstream run and 2 for a fatal error.

You can also resume from extracted passages using `--task3-input`. See [the input/output guide](docs/ARCHITECTURE.md) for the versioned contracts and output fields. A minimal Task 5-only installation is `python -m pip install -e .`; live collection and ranking need the full `requirements.txt` installation.

## How support is assessed

Task 5 checks citation traceability, question relevance, source/duplicate signals, OCR, coverage and unresolved contradictions. Generated answers also receive a criteria-based LLM review; NLI and sampling checks depend on the chosen profile and installed verifier.

**High / Medium / Low describe support from the supplied evidence. They are not factual accuracy percentages.** A well-supported complete answer can receive High. Missing coverage, unsupported claims and contradictions limit that label. Local Qwen self-review currently caps High proposals at Medium because the development pilot found overrating; Gemini-reviewed Qwen can reach High when the guards pass. The pilot used machine judgments; independent human calibration per language remains pending.

The website currently takes up to 24 ranked candidates from Task 4. LLM generation normally uses up to 4 diverse passages, with at most 2 per source group; a requested list can contain up to 8 supported points. CLI/API budgets can differ. These are prototype limits, not measured optimal settings.

## Tests

```sh
python scripts/check_release.py
python -m unittest discover -s task5/tests -q
python -m unittest discover -s website/tests -q
python -m pytest task1/tests task2/tests task4/tests pipeline/tests -q
node --test website/tests/ui.test.js
```

Tests use constructed fixtures, mock model responses, sanitised regression evidence and local HTTP sources. They do not require API keys or large models. Optional real-BGE tests are skipped by default. Passing these checks validates software behaviour, not answer quality on every query. CI runs the offline checks; hosted CI results are separate from local verification.

## Repository layout

```text
task1/       Query understanding and multilingual query generation
task2/       Web search, source collection and downloads
task3/       Document extraction, OCR and passage preparation
task4/       Evidence ranking, profiles, contracts and evaluation
task5/       One shared answer-generation and support-assessment package
pipeline/    Stage orchestration, live/resume CLI and integration tests
website/     Browser frontend, local HTTP backend and model setup tools
docs/        Setup, architecture, presentation guide and cleanup audit
scripts/     Source and release checks
```

| Folder | What to explain |
|---|---|
| [Task 1](task1/README.md) | Turn the question into search-query variants |
| [Task 2](task2/README.md) | Collect documents and preserve source metadata |
| [Task 3](task3/README.md) | Extract text/OCR and create evidence passages |
| [Task 4](task4/README.md) | Rank passages for the original question |
| [Task 5](task5/README.md) | Generate a cited answer and assess its support |
| [Pipeline](pipeline/README.md) | Connect the task interfaces and isolate each run |
| [Website](website/README.md) | Submit a question and inspect answers/sources |

Task 5 has one implementation, imported by both the website and pipeline; no sync script or mirrored code tree is needed. The [source manifest](docs/source_manifest.json) records preserved processing files and the path/configuration changes needed for this layout.

Search history, downloaded sources, logs, keys and weights stay local and are ignored by Git. This server binds to loopback and is intended for local use. It does not include authentication or a public multi-user deployment configuration.

See [troubleshooting and advanced setup](docs/LOCAL_SETUP.md) and [contributing notes](CONTRIBUTING.md).

## Acknowledgements

The project combines the team's Tasks 1–4 with Veeramalla Akshay's Task 5, pipeline integration and website work. [AllenAI Open-MDS](https://github.com/allenai/open-mds) was studied as a retrieval-based multi-document summarisation reference; Pramaan uses a temporary collection per query and its existing Task 4 interface. It does not install Open-MDS or require its trained checkpoints.

External libraries, model weights and source documents retain their respective licences and attribution. Regression fixtures are for software checks; constructed examples are marked as such.
