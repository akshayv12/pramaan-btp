"""
Task 3: English & Devanagari OCR & Text Preparation
====================================================
Main pipeline orchestrator. Implements the full extract_and_prepare() API contract.

Usage:
    python src/main.py                          # Batch process with Tesseract (baseline)
    python src/main.py --engine transformer     # Batch process with Surya Transformer OCR
"""
import os
import sys
import json
import argparse
from tempfile import TemporaryDirectory

from PIL import Image
import pymupdf as fitz  # PyMuPDF - used for page-level provenance

from router import route_document
from extractor import extract_text_direct
from ocr_baseline import run_ocr_with_confidence
from linguistics import detect_script, detect_language
from normalizer import normalize_text
from deduplicator import Deduplicator

# Optional: Transformer OCR engine (Surya)
try:
    from transformer_ocr import TransformerOCREngine, SURYA_AVAILABLE
except ImportError:
    SURYA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Passage Chunker: splits a wall of text into granular passages (~150-300 words)
# ---------------------------------------------------------------------------
def chunk_into_passages(text, max_words=250, overlap_words=30):
    """
    Splits extracted text into overlapping passages for downstream Task 4 
    vector embedding. Each passage is a semantically meaningful chunk.
    
    Strategy:
      1. Try to split on paragraph boundaries (double newlines) first.
      2. If a paragraph exceeds max_words, split it on sentence boundaries.
      3. Add a small word overlap between chunks for retrieval continuity.
    """
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    passages = []
    current_chunk = []
    current_word_count = 0
    
    for para in paragraphs:
        para_words = para.split()
        
        # If adding this paragraph would exceed the limit, flush the current chunk
        if current_word_count + len(para_words) > max_words and current_chunk:
            passage_text = '\n'.join(current_chunk)
            passages.append(passage_text)
            
            # Overlap: carry the last few words into the next chunk
            last_words = passage_text.split()[-overlap_words:]
            current_chunk = [' '.join(last_words)]
            current_word_count = len(last_words)
        
        current_chunk.append(para)
        current_word_count += len(para_words)
    
    # Don't forget the last chunk
    if current_chunk:
        passages.append('\n'.join(current_chunk))
    
    # If the text was too short to chunk, return it as a single passage
    if not passages and text.strip():
        passages = [text.strip()]
    
    return passages


def _recognize_image(image, transformer_engine, source_path=None):
    if transformer_engine is None:
        text, confidence, _ = run_ocr_with_confidence(image, lang="hin+eng")
        return text, confidence, "TESSERACT_BASELINE"

    if source_path is None:
        # The transformer takes a filename. Each PDF page gets a private temporary file.
        with TemporaryDirectory() as directory:
            page_path = os.path.join(directory, "page.png")
            image.save(page_path)
            text, confidence = transformer_engine.extract_document(page_path)
    else:
        text, confidence = transformer_engine.extract_document(source_path)
    confidence = confidence * 100 if confidence <= 1.0 else confidence
    return text, confidence, getattr(transformer_engine, "backend", "SURYA_TRANSFORMER")


def _document_pages(path, content_type, ocr_pages, transformer_engine):
    """Yield (page number, text, confidence percent, OCR engine) in reading order."""
    if content_type == "pdf":
        with fitz.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError("PDF requires a password")
            if ocr_pages is not None and any(
                type(n) is not int or not 1 <= n <= len(pdf) for n in ocr_pages
            ):
                raise ValueError("pages_needing_ocr must contain one-based pages in this PDF")
            hints = None if ocr_pages is None else set(ocr_pages)
            for number, page in enumerate(pdf, 1):
                native = page.get_text().strip()
                # Empty hints with unknown text-layer status are passed as None.
                needs_ocr = len(native) < 50 if hints is None else number in hints or not native
                if needs_ocr:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB, alpha=False)
                    with Image.frombytes("RGB", (pix.width, pix.height), pix.samples) as image:
                        yield number, *_recognize_image(image, transformer_engine)
                else:
                    yield number, native, 100.0, None
    elif content_type == "image":
        with Image.open(path) as image:
            yield 1, *_recognize_image(image, transformer_engine, path)
    else:
        yield None, extract_text_direct(path, content_type=content_type), 100.0, None


def extract_and_prepare(document_path, engine_type="tesseract",
                        transformer_engine=None, deduplicator=None, *,
                        document_id=None, content_type=None, ocr_pages=None):
    """Extract a saved document. Task 2 may supply an ID, media type and one-based PDF page hints."""
    document_path = os.fspath(document_path)
    filename = os.path.basename(document_path)
    doc_id = filename if document_id is None else document_id
    if content_type is None:
        if filename.lower().endswith(".pdf"):
            content_type = "pdf"
        else:
            content_type = "direct" if route_document(document_path) == "DIRECT_EXTRACTION" else "image"
    elif content_type not in {"html", "pdf", "image"}:
        raise ValueError(f"Unsupported document type: {content_type}")

    engine = transformer_engine if engine_type == "transformer" and SURYA_AVAILABLE else None
    print(f"--- Processing: {filename} ---")
    all_passages = []
    routes = set()
    for page, text, confidence, ocr_engine in _document_pages(document_path, content_type, ocr_pages, engine):
        routes.add("OCR_REQUIRED" if ocr_engine else "DIRECT_EXTRACTION")
        for idx, chunk in enumerate(chunk_into_passages(normalize_text(text)), 1):
            all_passages.append({
                "document_id": doc_id,
                "passage_id": f"{doc_id}_p{page or 1}_chunk{idx}",
                "source_file": filename,
                "source_path": document_path,
                "page_number": page,
                "extraction_method": "OCR" if ocr_engine else "DIRECT_EXTRACTION",
                "ocr_engine": ocr_engine,
                "detected_script": detect_script(chunk),
                "detected_language": detect_language(chunk),
                "ocr_confidence": round(confidence, 2),
                "reliability": "HIGH" if confidence >= 80 else ("MEDIUM" if confidence >= 60 else "LOW"),
                "text": chunk,
                "word_count": len(chunk.split()),
            })
    route = "MIXED_EXTRACTION" if len(routes) > 1 else next(iter(routes), "DIRECT_EXTRACTION")

    # Step 3: Near-duplicate removal via MinHash LSH
    unique_passages = []
    duplicates_removed = 0
    
    for passage in all_passages:
        if deduplicator:
            is_unique = deduplicator.process_document(
                passage["passage_id"], passage["text"]
            )
            if not is_unique:
                duplicates_removed += 1
                continue
        unique_passages.append(passage)
    
    # Step 4: Compute document-level summary statistics
    total_words = sum(p["word_count"] for p in unique_passages)
    avg_confidence = 0.0
    if unique_passages:
        conf_scores = [p["ocr_confidence"] for p in unique_passages]
        avg_confidence = round(sum(conf_scores) / len(conf_scores), 2)
    
    # Determine dominant script and language
    dominant_script = "Unknown"
    dominant_language = "Unknown"
    if unique_passages:
        from collections import Counter
        scripts = Counter(p["detected_script"] for p in unique_passages)
        languages = Counter(p["detected_language"] for p in unique_passages)
        dominant_script = scripts.most_common(1)[0][0]
        dominant_language = languages.most_common(1)[0][0]
    
    # Step 5: Package the final output
    result = {
        "document_id": doc_id,
        "source_path": document_path,
        "extraction_method": route,
        "ocr_engine": next((p["ocr_engine"] for p in unique_passages if p["ocr_engine"]), None),
        "dominant_script": dominant_script,
        "dominant_language": dominant_language,
        "total_passages": len(unique_passages),
        "duplicates_removed": duplicates_removed,
        "total_words": total_words,
        "avg_ocr_confidence": avg_confidence,
        "reliability": "HIGH" if avg_confidence >= 80 else ("MEDIUM" if avg_confidence >= 60 else "LOW"),
        "passages": unique_passages
    }
    
    print(f"  Engine: {result.get('ocr_engine', 'DIRECT')}")
    print(f"  Passages: {result['total_passages']} | Duplicates removed: {duplicates_removed}")
    print(f"  Script: {dominant_script} | Language: {dominant_language}")
    print(f"  Confidence: {avg_confidence}% ({result['reliability']})")
    
    return result


# ---------------------------------------------------------------------------
# Batch Processing CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: OCR & Text Preparation Pipeline")
    parser.add_argument("--engine", choices=["tesseract", "transformer"], default="tesseract",
                        help="OCR engine to use (default: tesseract)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing input documents (default: DATA_NEW/)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for JSON output (default: output/)")
    parser.add_argument("--dedup-threshold", type=float, default=0.85,
                        help="MinHash LSH similarity threshold for deduplication (default: 0.85)")
    args = parser.parse_args()
    
    # Resolve directories
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = args.data_dir or os.path.join(project_root, 'DATA_NEW')
    output_dir = args.output_dir or os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Initialize deduplicator
    dedup = Deduplicator(threshold=args.dedup_threshold)
    
    # Initialize transformer engine if requested
    transformer_engine = None
    if args.engine == "transformer":
        if SURYA_AVAILABLE:
            transformer_engine = TransformerOCREngine(languages=["hi", "en"])
        else:
            print("Warning: surya-ocr not installed. Falling back to Tesseract.")
            args.engine = "tesseract"
    
    print(f"Pipeline initialized:")
    print(f"  Engine: {args.engine.upper()}")
    print(f"  Data:   {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Dedup:  threshold={args.dedup_threshold}")
    print("=" * 60 + "\n")
    
    # Process all files
    all_results = []
    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.html', '.htm', '.pdf')
    
    for root, _, files in os.walk(data_dir):
        for filename in sorted(files):
            if not filename.lower().endswith(supported_exts):
                continue
            
            file_path = os.path.join(root, filename)
            
            try:
                result = extract_and_prepare(
                    file_path,
                    engine_type=args.engine,
                    transformer_engine=transformer_engine,
                    deduplicator=dedup
                )
                all_results.append(result)
                
                # Save individual JSON for Task 4 handoff
                json_filename = os.path.splitext(filename)[0] + "_extracted.json"
                json_path = os.path.join(output_dir, json_filename)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"  ERROR processing {filename}: {e}")
            
            print()
    
    # Write master summary
    summary = {
        "pipeline_engine": args.engine.upper(),
        "dedup_threshold": args.dedup_threshold,
        "total_documents": len(all_results),
        "total_passages": sum(r["total_passages"] for r in all_results),
        "total_duplicates_removed": sum(r["duplicates_removed"] for r in all_results),
        "total_words": sum(r["total_words"] for r in all_results),
        "documents": [
            {
                "file": r["document_id"],
                "script": r["dominant_script"],
                "language": r["dominant_language"],
                "passages": r["total_passages"],
                "confidence": r["avg_ocr_confidence"],
                "reliability": r["reliability"]
            }
            for r in all_results
        ]
    }
    
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print(f"  Documents processed: {summary['total_documents']}")
    print(f"  Total passages:      {summary['total_passages']}")
    print(f"  Duplicates removed:  {summary['total_duplicates_removed']}")
    print(f"  Total words:         {summary['total_words']}")
    print(f"  Output directory:    {output_dir}")
    print(f"  Summary:             {summary_path}")
