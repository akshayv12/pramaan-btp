"""Process the saved documents selected by a Task 2 v1.0 handoff with Smart 3-Lane Routing and Auto-Escalation."""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile


def _detect_devanagari_hint(doc):
    """
    Checks if a document or query has Devanagari/Hindi/Sanskrit hints,
    directing it to Lane 3 (Vision Transformer / Neural OCR) for superior accuracy.
    """
    instructions = doc.get("ocr_instructions", {})
    expected = [s.lower() for s in instructions.get("expected_scripts", [])]
    if any(s in expected for s in ["devanagari", "hindi", "sanskrit", "hi", "sa"]):
        return True

    meta = doc.get("source_metadata", {})
    lang = (meta.get("language") or "").lower()
    det_lang = (meta.get("detected_language") or "").lower()
    if lang in ["hi", "sa", "hin", "san"] or "devanagari" in det_lang or "hindi" in det_lang:
        return True

    orig_q = doc.get("original_query") or doc.get("query") or doc.get("search_query") or ""
    if any('\u0900' <= char <= '\u097f' for char in orig_q):
        return True

    title = (doc.get("title") or "").lower()
    if any('\u0900' <= char <= '\u097f' for char in title):
        return True

    return False


def _read_package(path):
    with path.open(encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    package = payload.get("task3_input") if isinstance(payload, dict) else None
    if not isinstance(package, dict) or package.get("schema_version") != "1.0":
        raise ValueError("expected task3_input with Task 2 schema_version '1.0'")
    documents = package.get("documents")
    queues = package.get("processing_requirements")
    if not isinstance(documents, list) or not isinstance(queues, dict) or not isinstance(package.get("metadata"), dict):
        raise ValueError("expected documents, processing_requirements and metadata in task3_input")

    ids = set()
    for doc in documents:
        key = doc.get("doc_id") if isinstance(doc, dict) else None
        if not isinstance(key, str) or not key.strip() or key in ids:
            raise ValueError(f"missing or duplicate Task 2 doc_id: {key!r}")
        ids.add(key)
    requests = {}
    for queue in ("ocr_required", "text_extracted"):
        values = queues.get(queue)
        if not isinstance(values, list):
            raise ValueError(f"missing or invalid processing list: {queue}")
        for key in values:
            if not isinstance(key, str) or key not in ids or key in requests:
                raise ValueError(f"unknown or repeated doc_id in {queue}: {key!r}")
            requests[key] = queue
    return package, requests


def _skip_reason(doc, queue):
    if doc.get("extraction_status") != "success":
        return doc.get("extraction_status") or "missing_extraction_status"
    for flag in ("robots_allowed", "meta_robots_allowed", "licence_permits_processing"):
        if doc.get(flag) is not True:
            return f"{flag} is not confirmed"
    return None if queue else "not_requested"


def _saved_file(value, root):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("selected document has no saved_path")
    # Task 2 also produces Windows-style relative paths when a handoff is moved.
    if os.name != "nt" and PureWindowsPath(value).drive:
        raise ValueError("saved_path is absolute on a different platform")
    path = (root / value.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("saved_path is outside the collection root")
    if not path.is_file():
        raise FileNotFoundError(f"saved document is missing: {value}")
    return path


def _write_output(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_manifest(input_path, output_path=None, *, collection_root=None, default_engine="auto"):
    """
    Write final passages and per-document outcomes.
    Fulfills 3-Lane Smart Routing and Auto-Escalation:
      - Lane 1: Direct extraction (HTML / Text PDFs)
      - Lane 2: Tesseract Baseline (Latin / Clean scans)
      - Lane 3: Vision Transformer / EasyOCR (Devanagari / Indic script priority)
      - Dynamic Auto-Escalation: Promotes low-confidence Tesseract scans (<60%) to Transformer
    """
    from main import extract_and_prepare
    from deduplicator import Deduplicator

    # Initialize Deep Transformer OCR Engine once for the batch if available
    transformer_engine = None
    try:
        from transformer_ocr import TransformerOCREngine, SURYA_AVAILABLE, EASYOCR_AVAILABLE
        if SURYA_AVAILABLE or EASYOCR_AVAILABLE:
            transformer_engine = TransformerOCREngine(languages=["hi", "en"])
            if transformer_engine.backend is None:
                transformer_engine = None
    except Exception as e:
        print(f"[TASK 3] Transformer OCR init notice: {e}", file=sys.stderr)
        transformer_engine = None

    transformer_ready = transformer_engine is not None and transformer_engine.backend is not None
    if transformer_ready:
        print(f"[TASK 3] Deep Vision Transformer ready: {transformer_engine.backend}")
    else:
        print("[TASK 3] Transformer not installed; running full baseline OCR.")

    manifest = Path(input_path).resolve()
    destination = Path(output_path).resolve() if output_path else manifest.with_name("task3_output.json")
    if destination == manifest:
        raise ValueError("output must differ from the Task 2 input manifest")
    root = Path(collection_root).resolve() if collection_root else manifest.parent
    package, requests = _read_package(manifest)
    deduplicators = defaultdict(Deduplicator)
    results = []
    
    for doc in package["documents"]:
        key = doc["doc_id"]
        queue = requests.get(key)
        # Task 2's 'text' can be a search snippet. The source record and extracted body stay distinct.
        item = {"document_id": key, "source": {k: v for k, v in doc.items() if k != "text"},
                "passages": [], "total_passages": 0}
        results.append(item)
        reason = _skip_reason(doc, queue)
        if reason:
            item.update(status="skipped", reason=reason)
            continue
        try:
            path = _saved_file(doc.get("saved_path"), root)
            expected_ocr = queue == "ocr_required"
            if doc.get("needs_ocr") is not expected_ocr:
                raise ValueError(f"needs_ocr disagrees with processing list {queue}")
            media = doc.get("content_type")
            if media not in {"html", "pdf", "image"}:
                raise ValueError(f"unsupported Task 2 content_type: {media!r}")
            if media != "pdf" and (media == "image") != expected_ocr:
                raise ValueError(f"Task 2 content_type {media!r} is inconsistent with {queue}")
            hints = None
            if media == "pdf":
                hints = (doc.get("ocr_instructions") or {}).get("pages_needing_ocr")
                if hints is not None and not isinstance(hints, list):
                    raise ValueError("pages_needing_ocr must be a list")
                if hints and not expected_ocr:
                    raise ValueError("direct-text PDF has conflicting pages_needing_ocr")
                if not hints and expected_ocr:
                    hints = None  # Task 2 could not inspect the PDF; let the extractor inspect every page.
            group = doc.get("query_slug") or doc.get("original_query") or ""

            # -------------------------------------------------------------
            # SMART 3-LANE ROUTING DECISION:
            # -------------------------------------------------------------
            if default_engine == "transformer" and transformer_ready:
                doc_engine = "transformer"
                lane_desc = "Lane 3: Vision Transformer (Manual Override)"
            elif default_engine == "tesseract":
                doc_engine = "tesseract"
                lane_desc = "Lane 2: Tesseract Baseline (Manual Override)"
            else: # "auto"
                if transformer_ready and _detect_devanagari_hint(doc):
                    doc_engine = "transformer"
                    lane_desc = "Lane 3: Vision Transformer (Devanagari / Indic script priority)"
                else:
                    doc_engine = "tesseract"
                    lane_desc = "Lane 2: Tesseract Baseline (Fast CPU Latin extraction)"

            if expected_ocr:
                print(f"  [ROUTER] {key} -> {lane_desc}")

            result = extract_and_prepare(
                path,
                engine_type=doc_engine,
                transformer_engine=transformer_engine,
                document_id=key,
                content_type=media,
                ocr_pages=hints,
                deduplicator=deduplicators[group]
            )

            # -------------------------------------------------------------
            # DYNAMIC AUTO-ESCALATION:
            # If Tesseract produced LOW reliability (< 60% confidence),
            # auto-escalate to Lane 3 (Transformer)
            # -------------------------------------------------------------
            if (doc_engine == "tesseract" and transformer_ready and
                result.get("extraction_method") in ["OCR", "OCR_REQUIRED", "MIXED_EXTRACTION"] and
                result.get("avg_ocr_confidence", 100.0) < 60.0):
                print(f"  [ROUTER] {key}: Low confidence ({result.get('avg_ocr_confidence')}%), auto-escalating to Lane 3 (Transformer)...")
                try:
                    escalated = extract_and_prepare(
                        path,
                        engine_type="transformer",
                        transformer_engine=transformer_engine,
                        document_id=key,
                        content_type=media,
                        ocr_pages=hints,
                        deduplicator=deduplicators[group]
                    )
                    if escalated.get("avg_ocr_confidence", 0) > result.get("avg_ocr_confidence", 0):
                        print(f"  [ROUTER] {key}: Escalation improved confidence to {escalated['avg_ocr_confidence']}% ({escalated['reliability']})")
                        result = escalated
                except Exception as esc_err:
                    print(f"  [ROUTER] {key}: Escalation fallback: {esc_err}", file=sys.stderr)

            result["source_path"] = doc["saved_path"]
            for passage in result["passages"]:
                passage["source_path"] = doc["saved_path"]
            item.update(result)
            if result["total_passages"]:
                item["status"] = "processed"
            elif result["duplicates_removed"]:
                item["status"] = "duplicate"
            else:
                item["status"] = "empty"
        except Exception as exc:
            # Third-party parsers and native OCR have different exception types.
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[TASK 3] {key}: {item['error']}", file=sys.stderr)

    counts = Counter(item["status"] for item in results)
    summary = {key: counts[key] for key in ("processed", "duplicate", "empty", "skipped", "failed")}
    summary.update(total_documents=len(results), total_passages=sum(r["total_passages"] for r in results))
    _write_output(destination, {"task3_output": {"schema_version": "1.0", "metadata": package["metadata"],
                                                "documents": results, "summary": summary}})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to task3_output.json beside the manifest")
    parser.add_argument("--collection-root", type=Path, help="Base for saved_path; defaults to the manifest directory")
    parser.add_argument("--engine", choices=["auto", "tesseract", "transformer"], default="auto",
                        help="OCR routing strategy: auto (Smart 3-Lane routing), tesseract (baseline), transformer")
    args = parser.parse_args(argv)
    output = args.output or args.manifest.with_name("task3_output.json")
    try:
        summary = process_manifest(args.manifest, output, collection_root=args.collection_root, default_engine=args.engine)
    except (ValueError, OSError, ImportError) as exc:
        print(f"[TASK 3] {exc}", file=sys.stderr)
        return 2
    print(f"[TASK 3] {summary} -> {output.resolve()}")
    return int(summary["failed"] > 0)


if __name__ == "__main__":
    sys.exit(main())
