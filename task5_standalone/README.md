# Task 5

The canonical implementation is in `task5/`; the integration copy is in `../integrated_pipeline/task5/`. Both include the support policy JSON and are checked for identical contents.

From the repository root, install with `python -m pip install -e .`, then try:

```sh
python task5_standalone/process_task4_output.py \
  task5_standalone/examples/contract_example.json \
  --provider extractive --output outputs/demo.json --text-output outputs/demo.txt
```

This command uses constructed evidence. It does not retrieve web documents or call a model. Real model setup and the full website are covered in the [root README](../README.md).

Task 5 takes the original question, ranked passages and source/OCR/context signals. It returns a source-grounded answer, claim-level citations, supporting sources and an interpretable support assessment. It preserves Task 4's IDs, scores and provenance rather than replacing its ranker. See [the input/output guide](../docs/ARCHITECTURE.md).

The regression fixtures are sanitised evidence for reproducing specific software failures. Their local path metadata has been replaced with portable placeholders. They are not a human-labelled factual benchmark or fresh search results. `contract_example.json` is locally authored and explicitly simulated.

After changing canonical code, run `python task5_standalone/sync_to_integrated.py` and the test commands in the root README. This copies Task 5 and its handoff tests only. It leaves Tasks 1–4 source files alone.
