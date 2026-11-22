# Task 5 — Grounded answer generation and support assessment

**Input:** the original question, Task 4 ranked passages and source/OCR/context signals.

**Work:** select diverse evidence, create exact source-statement cards, generate a concise answer using Gemini/local Qwen, bind claims to citations, check language/support/coverage, and explain the support label. Source extraction is the explicit no-LLM option.

**Output:** one multi-document answer, supporting sources/quotes and an interpretable **High / Medium / Low** assessment.

| File | Role |
|---|---|
| `handoff.py` | Validate/adapt Task 4 input and serialise results |
| `pipeline.py` | Connect selection, synthesis and assessment |
| `evidence.py`, `planning.py` | Diverse, question-focused context selection |
| `generation.py`, `synthesis.py`, `providers.py` | Source cards, drafting and model review |
| `nli.py`, `verification.py` | Optional NLI and retrieval/consistency diagnostics |
| `reliability.py`, `support.py`, `support_policy.json` | Support assessment and guards |
| `intent.py`, `language.py` | Requested language and answer format |
| `examples/`, `tests/` | Constructed/sanitised evidence and regression cases |

This folder is the **only Task 5 implementation**. The website and pipeline import it directly; there is no integrated copy or sync step. Original Task 4 scores, IDs and provenance remain intact.

From the repository root:

```sh
python -m task5 task5/examples/contract_example.json --provider extractive --output outputs/demo.json --text-output outputs/demo.txt
python -m unittest discover -s task5/tests -q
```

`btp-task5` is also available after `pip install -e .`. All entry points load the shared, ignored repository-root `.env`. See the [model setup guide](../README.md#choose-an-answer-model). An LLM is needed for translated synthesis; original quotes retain their language.

The website normally passes up to 24 candidates and generates from up to 4 selected passages, with at most 2 per source group. Verification profiles control NLI, review and sampling. Labels describe support from supplied evidence, not truth probabilities; independent human calibration remains pending.

**Presentation line:** “My Task 5 work converts ranked evidence into a concise cited answer and explains how strongly it is supported.”

See [architecture](../docs/ARCHITECTURE.md), [pipeline integration](../pipeline/README.md) and [the presentation guide](../docs/PRESENTATION_GUIDE.md).
