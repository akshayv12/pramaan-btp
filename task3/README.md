# Task 3 — Extraction, OCR and passage preparation

**Input:** Task 2's manifest and downloaded HTML/PDF/image files.

**Work:** choose direct extraction or OCR, normalise text, detect language/script, remove duplicate passages and retain source/page/extraction information. Its original subprocess boundary is preserved.

**Output:** `task3_output.json`: processed documents, passages, extraction/OCR metadata and the processed/failed/empty/duplicate summary.

| File | Role |
|---|---|
| `src/process_manifest.py` | Manifest-to-Task-3 handoff CLI |
| `src/main.py` | Document processing and passage preparation |
| `src/extractor.py` | Direct HTML/PDF extraction |
| `src/ocr_baseline.py` | Tesseract baseline |
| `src/router.py`, `src/transformer_ocr.py` | Existing extraction and optional OCR routes |
| `src/deduplicator.py`, `src/normalizer.py`, `src/detector.py` | Text preparation |
| `data/images/english/eng1.jpeg` | OCR fixture used by pipeline tests |

```sh
python task3/src/process_manifest.py PATH_TO_TASK3_INPUT.json --output outputs/task3_output.json
```

Saved input paths must resolve inside the collection root; use `--collection-root` when needed. The pipeline supplies run-specific paths. For scanned sources, install Tesseract and run `python website/setup_ocr.py`. Optional transformer/layout/Surya/EasyOCR routes and evaluation helpers remain available with their separate dependencies/weights.

Task 3's OCR percentage is converted to Task 4's unit scale in [the adapter](../task4/process_task3_output.py). Direct-text confidence is not treated as measured OCR quality.

**Presentation line:** “Task 3 turns collected documents into passages that later stages can quote and trace.” Next: [Task 4](../task4/README.md).
