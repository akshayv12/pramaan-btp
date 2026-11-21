"""Exercise the saved-file handoff and both supported pipeline entry points."""
import json
import shutil
import subprocess
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

import main as collector
import run_pipeline as runner

pymupdf = pytest.importorskip("pymupdf", reason="requires Task 3 baseline dependencies")
pytest.importorskip("datasketch", reason="requires Task 3 baseline dependencies")

TASK3 = Path(__file__).resolve().parents[1] / "Indian-Context-Retrieval-Summarization-task-3-baseline"
PROCESS = TASK3 / "src" / "process_manifest.py"
QUESTION = "What is the Green Revolution?"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def document(key, path="body.html", **changes):
    return {
        "doc_id": key, "saved_path": path, "content_type": "html", "needs_ocr": False,
        "extraction_status": "success", "robots_allowed": True, "meta_robots_allowed": True,
        "licence_permits_processing": True, "query_slug": "question",
        "url": f"https://example.org/{key}", "text": "Only a search result snippet.",
        **changes,
    }


def write_manifest(path, documents, **changes):
    requirements = {
        "ocr_required": [d["doc_id"] for d in documents if d["needs_ocr"]],
        "text_extracted": [d["doc_id"] for d in documents if not d["needs_ocr"]],
    }
    package = {"schema_version": "1.0", "metadata": {"original_query": QUESTION},
               "documents": documents, "processing_requirements": requirements, **changes}
    path.write_text(json.dumps({"task3_input": package}), encoding="utf-8")


def replay(path, *args):
    return subprocess.run([sys.executable, "-X", "utf8", "-B", str(PROCESS), str(path), *map(str, args)],
                          capture_output=True, encoding="utf-8", timeout=60)


@pytest.fixture
def local_collection(tmp_path, monkeypatch):
    web = tmp_path / "site"
    web.mkdir()
    (web / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")
    (web / "index.html").write_text(
        '<html><head><title>Agricultural history</title></head><body>'
        '<p>Harvest records were preserved in the village archive.</p>'
        '<a href="/green-revolution.pdf">Report</a></body></html>', encoding="utf-8")
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((30, 50), "The annual crop report describes changes in irrigation and seed varieties.")
        pdf.save(web / "green-revolution.pdf")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(web))) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}/index.html"
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("QUERY_BACKEND", "baseline")
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setattr(runner, "TASK1_OUTPUT", tmp_path / "task1_input.json")
        monkeypatch.setattr(collector, "GLOBAL_QUERY_DELAY", 0)
        monkeypatch.setattr(collector, "DEFAULT_DOMAIN_DELAY", 0)
        # Only the search provider is substituted. Task 2 fetches and parses both real files.
        monkeypatch.setattr(collector, "search_web", lambda *a, **kw: ([{"url": url, "text": ""}], "fixture"))
        try:
            yield url
        finally:
            server.shutdown()
            worker.join()


@pytest.mark.parametrize("full", [False, True], ids=["legacy_two_stage", "three_stage"])
def test_pipeline_entry_points(local_collection, tmp_path, full):
    run = runner.run_full_pipeline if full else runner.run_pipeline
    paths = run(QUESTION, max_queries=2, top_k=1)
    assert len(paths) == (3 if full else 2)
    assert read_json(paths[0])["original_query"] == QUESTION
    package = read_json(paths[1])["task3_input"]
    assert package["metadata"]["original_query"] == QUESTION
    source = package["documents"]
    assert [d["content_type"] for d in source] == ["html", "pdf"]
    assert all(d["text"] == "" and Path(d["saved_path"]).is_file() for d in source)
    assert package["processing_requirements"]["text_extracted"] == [d["doc_id"] for d in source]
    output = tmp_path / "task3_output.json"
    assert output.exists() == full
    if full:
        assert paths[2] == output
        final = read_json(output)["task3_output"]
        html, pdf = final["documents"]
        assert final["metadata"] == package["metadata"]
        assert final["summary"]["processed"] == 2
        assert html["source"]["url"] == local_collection
        assert pdf["source"]["parent_doc_id"] == html["document_id"]
        assert [r["document_id"] for r in (html, pdf)] == [d["doc_id"] for d in source]
        assert html["source"]["source_query_details"] == source[0]["source_query_details"]
        assert "village archive" in html["passages"][0]["text"]
        assert pdf["passages"][0]["page_number"] == 1
        assert "irrigation" in pdf["passages"][0]["text"]


def test_partial_output_retry_and_access_decisions(tmp_path):
    local = tmp_path / "collection"
    local.mkdir()
    path = tmp_path / "input.json"
    output = tmp_path / "result" / "passages.json"
    (local / "body.html").write_bytes("<p>भारत में खेती है. Document body.</p>".encode() + b"\xff")
    docs = [
        document("good"), document("missing", "missing.html"), document("blocked", robots_allowed=False),
        document("restricted", meta_robots_allowed=False), document("licence", licence_permits_processing=False),
        document("failed", extraction_status="fetch_failed"), document("snippet", extraction_status="snippet_only"),
        document("not_selected", "absent.html"), document("blank", "blank.html"),
    ]
    (local / "blank.html").write_text("<script>no body</script>", encoding="utf-8")
    write_manifest(path, docs, processing_requirements={"ocr_required": [],
                   "text_extracted": [d["doc_id"] for d in docs if d["doc_id"] != "not_selected"]})
    args = ("--output", output, "--collection-root", local)
    first = replay(path, *args)
    assert first.returncode == 1, first.stderr
    final = read_json(output)["task3_output"]
    by_id = {d["document_id"]: d for d in final["documents"]}
    assert [d["status"] for d in final["documents"]] == ["processed", "failed"] + ["skipped"] * 6 + ["empty"]
    assert "missing.html" in by_id["missing"]["error"]
    assert by_id["not_selected"]["reason"] == "not_requested"
    passage = by_id["good"]["passages"][0]
    assert "भारत में खेती है" in passage["text"] and "snippet" not in passage["text"]
    assert passage["document_id"] == "good" and passage["source_path"] == "body.html"
    (local / "missing.html").write_text("<p>Restored article on local administration.</p>", encoding="utf-8")
    second = replay(path, *args)
    assert second.returncode == 0, second.stderr
    assert read_json(output)["task3_output"]["summary"] == {
        "processed": 2, "duplicate": 0, "empty": 1, "skipped": 6, "failed": 0,
        "total_documents": 9, "total_passages": 2,
    }


def test_distinct_document_ids_and_query_scoped_deduplication(tmp_path):
    docs = []
    for key, text, group in (("a", "Local archive", "one"), ("b", "Local archive", "one"),
                             ("c", "Seed bank", "one"), ("d", "Local archive", "two")):
        folder = tmp_path / key
        folder.mkdir()
        (folder / "page.html").write_text(text, encoding="utf-8")
        docs.append(document(key, f"{key}\\page.html", query_slug=group))
    path = tmp_path / "input.json"
    write_manifest(path, docs)
    run = replay(path)
    assert run.returncode == 0, run.stderr
    result = read_json(tmp_path / "task3_output.json")["task3_output"]["documents"]
    assert [d["status"] for d in result] == ["processed", "duplicate", "processed", "processed"]
    ids = [p["passage_id"] for d in result for p in d["passages"]]
    assert ids == ["a_p1_chunk1", "c_p1_chunk1", "d_p1_chunk1"]


@pytest.mark.parametrize("bad", ["version", "unknown_id", "duplicate_id", "ambiguous_queue"])
def test_invalid_manifest_preserves_last_output(tmp_path, bad):
    path = tmp_path / "input.json"
    output = tmp_path / "task3_output.json"
    output.write_bytes(b"previous completed result")
    docs = [document("one")]
    changes = {}
    if bad == "version":
        changes["schema_version"] = "9"
    elif bad == "duplicate_id":
        docs.append(document("one"))
    else:
        changes["processing_requirements"] = {"ocr_required": ["unknown" if bad == "unknown_id" else "one"],
                                              "text_extracted": ["one"]}
    write_manifest(path, docs, **changes)
    assert replay(path).returncode == 2
    assert output.read_bytes() == b"previous completed result"


def test_path_escape_and_route_mismatch_are_reported(tmp_path):
    collection = tmp_path / "collection"
    collection.mkdir()
    (tmp_path / "outside.html").write_text("outside", encoding="utf-8")
    (collection / "body.html").write_text("inside", encoding="utf-8")
    path = collection / "input.json"
    docs = [document("outside", "../outside.html"), document("mismatch"),
            document("image", content_type="image"),
            document("pdf", content_type="pdf", ocr_instructions={"pages_needing_ocr": [1]})]
    write_manifest(path, docs, processing_requirements={
        "ocr_required": ["mismatch"], "text_extracted": ["outside", "image", "pdf"]})
    assert replay(path).returncode == 1
    results = read_json(collection / "task3_output.json")["task3_output"]["documents"]
    assert all(result["status"] == "failed" for result in results)
    assert "outside the collection root" in results[0]["error"]
    assert "needs_ocr" in results[1]["error"]
    assert "content_type" in results[2]["error"]
    assert "pages_needing_ocr" in results[3]["error"]


@pytest.mark.skipif(not shutil.which("tesseract") and not (Path(sys.prefix) / "ocr/tesseract.exe").exists(),
                    reason="requires the native Tesseract runtime")
def test_task2_page_hints_and_real_ocr(tmp_path):
    pdf_path = tmp_path / "mixed.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((30, 50), "The digital first page describes Indian agricultural production in detail.")
        page = pdf.new_page(width=447, height=386)
        page.insert_image(page.rect, filename=str(TASK3 / "data/images/english/eng1.jpeg"))
        pdf.save(pdf_path)
    native, ocr = collector.inspect_pdf_text_pages(pdf_path)
    assert (native, ocr) == ([1], [2])
    record = document("mixed", "mixed.pdf", content_type="pdf", needs_ocr=True,
                      ocr_instructions={"pages_needing_ocr": ocr})
    path = tmp_path / "input.json"
    write_manifest(path, [record])
    run = replay(path)
    assert run.returncode == 0, run.stderr
    result = read_json(tmp_path / "task3_output.json")["task3_output"]["documents"][0]
    first, second = result["passages"]
    assert result["extraction_method"] == "MIXED_EXTRACTION"
    assert (first["page_number"], first["extraction_method"]) == (1, "DIRECT_EXTRACTION")
    assert (second["page_number"], second["ocr_engine"]) == (2, "TESSERACT_BASELINE")
    assert "pleasant" in second["text"].lower() and 0 < second["ocr_confidence"] <= 100


def test_upstream_failure_stops_task3_and_protects_handoffs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "TASK1_OUTPUT", tmp_path / "task1_input.json")
    with pytest.raises(ValueError, match="max_queries"):
        runner.run_full_pipeline(QUESTION, max_queries=1)
    with pytest.raises(ValueError, match="handoffs"):
        runner.run_full_pipeline(QUESTION, output=tmp_path / "task1_input.json")
    assert list(tmp_path.iterdir()) == []


def test_empty_collection_and_cli_failure_propagation(tmp_path, monkeypatch):
    path = tmp_path / "task3_input.json"
    output = tmp_path / "task3_output.json"
    write_manifest(path, [])
    ok = replay(path)
    assert ok.returncode == 0, ok.stderr
    assert read_json(output)["task3_output"]["summary"]["total_documents"] == 0

    write_manifest(path, [document("missing", "unavailable.html")])
    monkeypatch.setattr(runner, "run_pipeline", lambda *args: (tmp_path / "task1.json", path))
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--with-task3", "--query", QUESTION])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 1
    assert path.exists() and read_json(output)["task3_output"]["summary"]["failed"] == 1
