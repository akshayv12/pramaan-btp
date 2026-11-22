# Presentation walkthrough

Each task folder's README gives its input, work, output and main files. Open them in processing order:

| Order | Folder / main file | Explain |
|---|---|---|
| 1 | [Task 1](../task1/README.md), `app.py` | Question → multilingual search variants |
| 2 | [Task 2](../task2/README.md), `main.py` | Variants → collected documents and source manifest |
| 3 | [Task 3](../task3/README.md), `src/process_manifest.py` | Documents → passages with extraction/OCR metadata |
| 4 | [Task 4](../task4/README.md), `task4_rank_evidence.py` | Passages → ranked evidence with relevance/context/source signals |
| 5 | [Task 5](../task5/README.md), `pipeline.py` | Ranked evidence → cited answer and explained support |
| 6 | [Pipeline](../pipeline/README.md) | Live/resume modes, stage interfaces, isolated runs and failure handling |
| 7 | [Website](../website/README.md) | Query → answer, sources, language and verification details |

## Your contribution

Focus the detailed explanation on **Task 5, pipeline integration and the website**. Acknowledge the team's Tasks 1–4.

For Task 5, show the original question and ranked passages. Explain diverse selection, source-statement cards, Gemini/Qwen drafting and claim citations. Then show coverage, source/OCR signals, NLI/review status and the High/Medium/Low explanation. The label describes evidence support, not a truth percentage.

For integration, trace one run folder: `task1_input.json → task3_input.json → task3_output.json → task4_output.json → task5_output.json`. Show how uploading Task 4 output bypasses collection and ranking, helping isolate generation problems.

For the website, show the query, selected model and language, then open sources and verification details. The frontend submits a job to the local Python backend, which delegates to the same pipeline and single Task 5 package.

## Suggested demo

1. Run the authored Hindi saved example using Source extraction, showing the interface without an external model.
2. With your model configured, ask `Tell me about Sundara Kanda`. Inspect the stage trace and source quotes.
3. Request Hindi explicitly and demonstrate synthesis in the requested language. Source extraction keeps source language.
4. Reuse the resulting Task 4 JSON and show that ranking is supplied rather than repeated.
5. Show a support explanation and limited-evidence case, keeping verification limitations visible.

Live results depend on available evidence and services; this is a suggested walkthrough, not a claim about every result. Keep keys, private history and academic reports outside the code view.
