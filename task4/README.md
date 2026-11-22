# Task 4 — Indian-context evidence ranking

**Input:** the original question and prepared passages with source/extraction metadata.

**Work:** score query relevance, Indian-context relevance and source signals, then rank passages. The lexical profile uses the existing TF-IDF method; semantic ranking also uses BGE-M3. Metadata scores are not independent factual verification.

**Output:** `task4_output.json`, with ranks, original passage IDs/text, component/final scores, provenance and references for Task 5.

| File / folder | Role |
|---|---|
| `task4_rank_evidence.py` | Ranker and configuration validation |
| `context_features.py`, `evidence_groups.py` | Context and source/duplicate signals |
| `process_task3_output.py` | Task 3 → Task 4 adaptation and batch grouping |
| `contracts.py`, `schemas/` | Versioned input/output validation |
| `configs/` | Semantic and distinct lexical profiles |
| `evaluation/`, `tests/` | Metrics, experiments, annotation tools and regression tests |
| `scripts/prepare_model.py` | Explicit optional BGE-M3 setup |

```sh
python task4/process_task3_output.py PATH_TO_TASK3_OUTPUT.json \
  --lexical-only --output outputs/task4_output.json
python -m pytest task4/tests -q
```

The website requests up to 24 candidates. `integration_lexical.json` retains the original relevance gate; `website_lexical.json` disables it so Task 5 can inspect weak lexical matches. These profiles are not duplicates. Production weights and filtering values are unchanged.

`examples/integration_demo.py` checks the Task 4 contract with a stub consumer. The real Task 5 flow is covered by `pipeline/tests/test_task5_pipeline.py`. See [optional BGE setup](../docs/LOCAL_SETUP.md#semantic-ranking-with-bge-m3).

**Presentation line:** “Task 4 chooses relevant evidence passages; Task 5 uses those passages to write the answer.” Next: [Task 5](../task5/README.md).
