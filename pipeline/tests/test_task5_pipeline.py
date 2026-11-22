"""Run the original Tasks 1–4 code, then the real Task 5 consumer.

Only the search result list is redirected to local HTML/PDF test documents.
No fake ranker, extractor, summarizer or reliability assessor is used.
"""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import run_pipeline as runner
from test_task3_pipeline import QUESTION, document, local_collection, read_json, write_manifest


@pytest.fixture
def lexical_config(tmp_path):
    path=tmp_path/"config.json"
    path.write_text(json.dumps({"relevance_weights":{"lexical_weight":1.,"semantic_weight":0.},
                               "output":{"include_source_support":True}}))
    return path


def test_real_five_stage_pipeline(local_collection,tmp_path,lexical_config,monkeypatch):
    path=tmp_path/"site"/"index.html"
    path.write_text(path.read_text().replace('</body>',"<p>The Green Revolution increased wheat production through improved seed varieties and irrigation.</p></body>"))
    monkeypatch.setenv("TASK4_PYTHON",sys.executable)
    paths=runner.run_task5_pipeline(QUESTION,max_queries=2,top_k=1,task4_config=lexical_config,task5_provider="extractive")
    assert len(paths)==5
    for p in paths:assert Path(p).is_file()
    output=read_json(paths[-1])["task5_output"]
    [item]=output["queries"]
    assert item["integration"]["actual_task5_available"] is True
    assert item["result"]["claims"]
    assert item["query"]==QUESTION
    ranked=read_json(paths[3])["task4_output"]["queries"][0]["ranked_evidence"]
    originals={r["passage_id"]:r for r in ranked}
    for p in item["result"]["used_passages"]:
        original=originals[item["citation_to_task4_passage_id"][p["id"]]]
        assert p["text"]==original["text"]
        assert p["query_rel"]==original["query_relevance"]
        assert p["url_or_provenance"].startswith("http://127.0.0.1:")
    assert output["task3_summary"]["processed"]==2


def test_partial_failure_still_reaches_task5_and_cli_exits_one(tmp_path,monkeypatch,lexical_config):
    monkeypatch.chdir(tmp_path)
    (tmp_path/"body.html").write_text("<p>The Green Revolution increased wheat production in Punjab through new seeds.</p>")
    manifest=tmp_path/"task3_input.json"
    write_manifest(manifest,[document("good"),document("bad","missing.html")])
    monkeypatch.setenv("TASK4_PYTHON",sys.executable)
    monkeypatch.setattr(runner,"run_pipeline",lambda *a:(tmp_path/"task1.json",manifest))
    monkeypatch.setattr(sys,"argv",["run_pipeline.py","--with-task5","--task4-config",str(lexical_config),
                                   "--task5-provider","extractive","--query",QUESTION])
    with pytest.raises(SystemExit) as error:runner.main()
    assert error.value.code==1
    [answer]=read_json(tmp_path/"task5_output.json")["task5_output"]["queries"]
    assert answer["result"]["claims"]
    assert any("failed extraction" in w for w in answer["result"]["warnings"])


def test_fatal_task3_error_stops_task5(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source=tmp_path/"task3_input.json";source.write_text('{"task3_input":{"schema_version":"9"}}')
    monkeypatch.setattr(runner,"run_pipeline",lambda *a:(tmp_path/"task1.json",source))
    with pytest.raises(subprocess.CalledProcessError):runner.run_task5_pipeline(QUESTION,task5_provider="extractive")
    assert not (tmp_path/"task5_output.json").exists()


def test_overwriting_upstream_handoffs_is_rejected_before_collection(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner,"TASK1_OUTPUT",tmp_path/"task1_input.json")
    for name in ("task1_input.json","task3_input.json","task3_output.json","task4_output.json"):
        with pytest.raises(ValueError,match="four upstream"):
            runner.run_task5_pipeline(QUESTION,task5_output=tmp_path/name)
    assert not list(tmp_path.iterdir())
