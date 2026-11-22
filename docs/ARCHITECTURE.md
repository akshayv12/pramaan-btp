# Architecture and input/output

## Pipeline responsibilities

| Stage | Responsibility | Handoff |
|---|---|---|
| Task 1 | Query understanding and multilingual query variants | `task1_input.json` |
| Task 2 | Existing search cascade, URL collection and download manifest | `task3_input.json` |
| Task 3 | HTML/PDF extraction, optional OCR, normalisation and passages | `task3_output.json` |
| Task 4 | Query/context/source scoring and ranked evidence | `task4_output.json` |
| Task 5 | Diverse context selection, grounded answer generation and support review | `task5_output.json` |

The unified CLI and website use the existing subprocess boundaries so upstream modules named `app` or `main` do not collide. A live run has its own directory. The browser polls the Python backend for stage status; the backend delegates research to `pipeline/run_integrated_pipeline.py`.

## Task 4 input to Task 5

Task 5 accepts a single-query envelope with `schema_version: "1.0.0"`, or a batch:

```json
{
  "task4_output": {
    "schema_version": "1.0",
    "queries": ["ONE_OR_MORE_SINGLE_QUERY_ENVELOPES"]
  }
}
```

The string in this structural illustration stands for an actual envelope object. Use [the complete constructed envelope](../task5/examples/contract_example.json) as a valid input, rather than this abbreviated illustration. The formal contracts are [Task 3 → 4](../task4/schemas/task3_to_task4.v1.schema.json) and [Task 4 → 5](../task4/schemas/task4_to_task5.v1.schema.json).

Each ranked item carries `rank`, `passage_id`, `text`, `query_relevance`, `indian_context_relevance`, `source_signal`, `final_score` and provenance. Source metadata can include URL/title/document ID/type; extraction metadata carries page/method and optional OCR confidence. Traceability contains the matching selected IDs and references. Malformed envelopes are rejected.

Task 5 assigns readable citation IDs such as `S1`, tied to the supplied rank, and retains a mapping back to the original passage IDs. It preserves the original scores and text. Scores are signals from Task 4, not probabilities of correctness.

## Example flow

Suppose Task 4 returns these passages for `How can researchers use these manuscript collections?`:

| Rank / citation | Passage | Source |
|---|---|---|
| 1 / S1 | The archive provides a public catalogue of manuscripts. Researchers can request reading-room access. | Constructed archive fixture |
| 2 / S2 | The library helps researchers locate manuscript collections. Staff provide catalogue guidance and reference assistance. | Constructed library fixture |

Task 5 first checks usable content and source repetition, then selects evidence. Source extraction quotes supported statements; an LLM can phrase a concise answer such as:

> Researchers can use the public catalogue and request reading-room access [S1]. Library staff can help locate collections and guide catalogue searches [S2].

This is an illustration using authored fixture text. Actual wording and the support label come from the run. The output includes the source URLs/quotes and a separate explanation of support; simulated sources do not establish real-world factual reliability.

## Task 5 output

The top-level `task5_output` contains a `queries` list. Each query result includes:

- `query`, `answer_status` and `display` for the requested answer.
- `result.answer`, `result.target_language` and the actual `generation_mode`.
- `result.claims`, `result.supporting_sources` and `result.cited_sources`.
- `result.reliability`, `result.reliability_explanation` and interpretable `result.factors`.
- `result.used_passages`, warnings and verification/model-call diagnostics.
- `citation_to_task4_passage_id`, so a visible citation can be traced to Task 4.

The unified runner records execution mode (`live_tasks1_to_5`, `resume_task3`, or `resume_task4`). Exact answer text and optional diagnostics depend on the provider/profile. The no-key CLI example in the README creates a complete JSON output you can inspect locally.

## Generation and verification

Task 5 selects a diverse context from the ranked candidates, creates exact source-statement cards with IDs, and asks the selected provider for cited claims. The server binds those claims to known evidence rather than letting the model invent source records. Content, numeric/qualifier, citation and language checks remain in the generation path.

Standard verification adds local multilingual NLI where installed and a criteria-based model review. Review considers faithfulness, question relevance, coverage, source quality, agreement and language/readability. Thorough adds a sampled alternative and a consistency signal. Missing NLI, contradictions, incomplete coverage and review failures are surfaced.

The support policy JSON is distributed with the single Task 5 package and its installable wheel. The website and pipeline import that same package. Its machine-judged pilot metadata is preserved; historical run logs and reports are excluded. Human calibration remains pending.

## Local HTTP API

| Method / path | Use |
|---|---|
| `GET /api/health` | Backend/component readiness |
| `POST /api/search` | Submit `{ "query": "...", "provider": "gemini", "language": "hi", "ranker": "lexical" }` |
| `POST /api/example` | Run the saved example; select `provider: "extractive"` for no-key use |
| `POST /api/task4` | Submit `{ "payload": TASK4_OBJECT, "provider": "extractive" }` |
| `GET /api/jobs` | Local history |
| `GET /api/jobs/{id}` | Poll status, stages and answers |
| `POST /api/jobs/{id}/cancel` | Cancel a queued/running job |
| `GET /api/jobs/{id}/json`, `/text`, `/log` | Download this job's output |
| `POST /api/gemini/configure` | Validate session model/key configuration |

Requests use `application/json` and the same local origin as the frontend. The backend validates request sizes, local Host/Origin, provider availability and handoff contracts. It serves allowlisted assets/results and renders source/model text as text with HTTP(S)-only links.

## Release preservation

[source_manifest.json](source_manifest.json) records unchanged processing files and the path/configuration edits required by the task folders. `scripts/check_release.py` checks the recorded hashes and the single Task 5 package. Query generation, collection, extraction, ranking weights, evidence selection and synthesis/reliability rules are preserved. Stage imports/subprocess paths, the verifier directory and shared environment loading follow the new layout. Regression fixtures retain their evidence and scores. The saved browser example is explicitly constructed. See [the cleanup audit](CLEANUP_AUDIT.md) for duplicate verification and the rationale for retained files.
