"""Exercise the Task 3 -> Task 4 handoff through the pipeline runner.

Task 4 runs in the root environment (PyTorch and BGE-M3). These tests pass a
lexical-only config so no encoder is loaded; ranking quality is covered by the
Task 4 suite.
"""
import json
import sys
from pathlib import Path

import pytest

import run_pipeline as runner
from test_task3_pipeline import QUESTION, document, local_collection, read_json, write_manifest  # noqa: F401

@pytest.fixture
def lexical_config(tmp_path):
    path = tmp_path / "task4_config.json"
    path.write_text(json.dumps({"relevance_weights": {"lexical_weight": 1.0, "semantic_weight": 0.0}}),
                    encoding="utf-8")
    return path


def test_four_stage_pipeline(local_collection, tmp_path, lexical_config):
    paths = runner.run_task4_pipeline(QUESTION, max_queries=2, top_k=1, task4_config=lexical_config, task4_top_k=5)
    task1, manifest, task3, task4 = paths
    assert read_json(task1)["original_query"] == QUESTION
    assert task3 == tmp_path / "task3_output.json" and task4 == tmp_path / "task4_output.json"
    extracted = read_json(task3)["task3_output"]
    passages = {p["passage_id"]: p for d in extracted["documents"] for p in d["passages"]}
    result = read_json(task4)["task4_output"]
    assert result["metadata"] == extracted["metadata"]
    assert result["task3_summary"] == extracted["summary"]
    [ranked] = result["queries"]
    assert ranked["query"] == QUESTION and ranked["status"] == "ok"
    assert ranked["candidate_count"] == len(passages)
    rows = ranked["ranked_evidence"]
    assert rows and [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    for row in rows:
        original = passages[row["passage_id"]]
        source = row["provenance"]["source_metadata"]
        assert source["document_id"] == original["document_id"]
        assert source["url"].startswith("http://127.0.0.1:")
    pdf = next(r for r in rows if r["provenance"]["source_metadata"]["content_type"] == "pdf")
    assert pdf["provenance"]["extraction_metadata"]["page_number"] == 1
    assert ranked["traceability"]["selected_evidence_ids"] == [r["passage_id"] for r in rows]


def test_partial_task3_failure_still_ranks_and_cli_exits_1(tmp_path, monkeypatch, lexical_config):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "body.html").write_text("<p>The Green Revolution raised wheat yields in Punjab.</p>",
                                        encoding="utf-8")
    manifest = tmp_path / "task3_input.json"
    write_manifest(manifest, [document("good"), document("missing", "missing.html")])
    monkeypatch.setattr(runner, "run_pipeline", lambda *args: (tmp_path / "task1.json", manifest))
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--with-task4", "--task4-config", str(lexical_config),
                                      "--query", QUESTION])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 1
    assert read_json(tmp_path / "task3_output.json")["task3_output"]["summary"]["failed"] == 1
    [ranked] = read_json(tmp_path / "task4_output.json")["task4_output"]["queries"]
    assert [r["passage_id"] for r in ranked["ranked_evidence"]] == ["good_p1_chunk1"]
    assert ranked["provenance"]["document_status_counts"]["failed"] == 1


def test_fatal_task3_error_stops_before_task4(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "task3_input.json"
    manifest.write_text(json.dumps({"task3_input": {"schema_version": "9"}}), encoding="utf-8")
    monkeypatch.setattr(runner, "run_pipeline", lambda *args: (tmp_path / "task1.json", manifest))
    with pytest.raises(runner.subprocess.CalledProcessError) as error:
        runner.run_task4_pipeline(QUESTION)
    assert error.value.returncode == 2
    assert not (tmp_path / "task4_output.json").exists()


def test_invalid_task4_arguments_fail_before_collection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "TASK1_OUTPUT", tmp_path / "task1_input.json")
    with pytest.raises(ValueError, match="three upstream"):
        runner.run_task4_pipeline(QUESTION, output=tmp_path / "task3_output.json")
    with pytest.raises(ValueError, match="two upstream"):
        runner.run_task4_pipeline(QUESTION, task3_output=tmp_path / "task1_input.json")
    with pytest.raises(ValueError, match="task4_top_k"):
        runner.run_task4_pipeline(QUESTION, task4_top_k=-1)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--task4-output", "x.json", "--query", QUESTION])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
