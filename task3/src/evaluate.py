"""
Task 3: Formal Evaluation Suite
================================
Compares Tesseract (Classical Baseline) vs Surya Transformer OCR 
using CER and WER metrics against ground-truth transcriptions.

Usage:
    python src/evaluate.py                    # Evaluate Tesseract only
    python src/evaluate.py --transformer      # Evaluate both engines side-by-side
"""
import os
import sys
import time
import json
import argparse
from PIL import Image

from jiwer import cer, wer
from ocr_baseline import _process_image_3path, run_ocr_with_confidence
from normalizer import normalize_text
from linguistics import detect_script

# Optional transformer engine
TRANSFORMER_AVAILABLE = False
try:
    from transformer_ocr import TransformerOCREngine, SURYA_AVAILABLE, EASYOCR_AVAILABLE
    TRANSFORMER_AVAILABLE = SURYA_AVAILABLE or EASYOCR_AVAILABLE
except ImportError:
    SURYA_AVAILABLE = False
    EASYOCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Ground-truth test data (inline for portability)
# ---------------------------------------------------------------------------
# Each entry: (image_path_relative_to_data/, ground_truth_text, language, script)
# We will also check if a .gt.txt file exists alongside the image for custom ground truth.

BUILTIN_TESTS = [
    {
        "id": "eng1",
        "path": "data/images/english/eng1.jpeg",
        "language": "English",
        "script": "Latin",
        "ground_truth": None
    },
    {
        "id": "eng2", 
        "path": "data/images/english/eng2.jpeg",
        "language": "English",
        "script": "Latin",
        "ground_truth": None
    },
    {
        "id": "hin1",
        "path": "data/images/hindi/hin1.jpeg",
        "language": "Hindi",
        "script": "Devanagari",
        "ground_truth": None
    },
    {
        "id": "hin2",
        "path": "data/images/hindi/hin2.jpeg",
        "language": "Hindi",
        "script": "Devanagari",
        "ground_truth": None
    },
    {
        "id": "hin4",
        "path": "data/images/hindi/hin4.jpeg",
        "language": "Hindi",
        "script": "Devanagari",
        "ground_truth": None
    },
    {
        "id": "san1",
        "path": "data/images/sanskrit/san1.jpeg",
        "language": "Sanskrit",
        "script": "Devanagari",
        "ground_truth": None
    },
    {
        "id": "san2",
        "path": "data/images/sanskrit/san2.jpeg",
        "language": "Sanskrit",
        "script": "Devanagari",
        "ground_truth": None
    },
]


def load_ground_truth(image_path, project_root):
    """
    Looks for a ground-truth file alongside the image.
    Checks for: <image_name>.gt.txt  (e.g., eng1.gt.txt next to eng1.jpeg)
    """
    base = os.path.splitext(os.path.join(project_root, image_path))[0]
    gt_path = base + ".gt.txt"
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def run_tesseract_on_image(image_path):
    """Runs Tesseract 3-path OCR on a single image and returns (text, confidence, latency)."""
    img = Image.open(image_path)
    
    start = time.time()
    text = _process_image_3path(img)
    latency = round(time.time() - start, 2)
    
    # Get confidence
    _, conf, reliability = run_ocr_with_confidence(img, lang='hin+eng')
    
    text = normalize_text(text)
    return text, conf, latency


def run_transformer_on_image(image_path, engine):
    """Runs Surya Transformer OCR on a single image and returns (text, confidence, latency)."""
    start = time.time()
    text, conf = engine.extract_document(image_path)
    latency = round(time.time() - start, 2)
    
    text = normalize_text(text)
    # Surya returns confidence as 0.0-1.0, convert to 0-100 scale
    conf_pct = conf * 100 if conf <= 1.0 else conf
    return text, conf_pct, latency


def evaluate_engine(test_cases, project_root, engine_name, run_fn):
    """
    Evaluates an OCR engine against ground-truth test cases.
    Returns a list of result dicts.
    """
    results = []
    
    for test in test_cases:
        image_path = os.path.join(project_root, test["path"])
        
        if not os.path.exists(image_path):
            print(f"  [SKIP] {test['id']}: Image not found at {image_path}")
            continue
        
        # Load ground truth
        gt = test.get("ground_truth") or load_ground_truth(test["path"], project_root)
        if not gt:
            print(f"  [SKIP] {test['id']}: No ground-truth file found (create {test['id']}.gt.txt)")
            continue
        
        # Run OCR
        try:
            ocr_text, confidence, latency = run_fn(image_path)
        except Exception as e:
            print(f"  [ERROR] {test['id']}: {e}")
            continue
        
        # Calculate error rates
        if ocr_text.strip() and gt.strip():
            test_cer = round(cer(gt, ocr_text) * 100, 2)
            test_wer = round(wer(gt, ocr_text) * 100, 2)
        else:
            test_cer = 100.0
            test_wer = 100.0
        
        result = {
            "id": test["id"],
            "engine": engine_name,
            "language": test["language"],
            "script": test["script"],
            "cer_percent": test_cer,
            "wer_percent": test_wer,
            "confidence": round(confidence, 2),
            "latency_sec": latency,
            "ground_truth_preview": gt[:80] + "..." if len(gt) > 80 else gt,
            "ocr_output_preview": ocr_text[:80] + "..." if len(ocr_text) > 80 else ocr_text,
        }
        results.append(result)
        
        print(f"  [{test['id']}] CER={test_cer}% | WER={test_wer}% | Conf={confidence:.1f}% | {latency}s")
    
    return results


def print_comparison_table(all_results):
    """Prints a formatted comparison table of all results."""
    if not all_results:
        print("\nNo results to display. Create .gt.txt ground-truth files for your test images.")
        return
    
    print("\n" + "=" * 90)
    print(f"{'Engine':<22} {'Test':<8} {'Lang':<12} {'Script':<12} {'CER%':<8} {'WER%':<8} {'Conf%':<8} {'Time':<6}")
    print("-" * 90)
    
    for r in all_results:
        print(f"{r['engine']:<22} {r['id']:<8} {r['language']:<12} {r['script']:<12} "
              f"{r['cer_percent']:<8} {r['wer_percent']:<8} {r['confidence']:<8} {r['latency_sec']:<6}")
    
    print("=" * 90)
    
    # Per-script averages
    from collections import defaultdict
    by_engine_script = defaultdict(lambda: {"cer": [], "wer": [], "conf": [], "lat": []})
    
    for r in all_results:
        key = (r["engine"], r["script"])
        by_engine_script[key]["cer"].append(r["cer_percent"])
        by_engine_script[key]["wer"].append(r["wer_percent"])
        by_engine_script[key]["conf"].append(r["confidence"])
        by_engine_script[key]["lat"].append(r["latency_sec"])
    
    print(f"\n{'Engine':<22} {'Script':<12} {'Avg CER%':<10} {'Avg WER%':<10} {'Avg Conf%':<10} {'Avg Time':<8}")
    print("-" * 72)
    
    for (engine, script), vals in sorted(by_engine_script.items()):
        avg_cer = round(sum(vals["cer"]) / len(vals["cer"]), 2)
        avg_wer = round(sum(vals["wer"]) / len(vals["wer"]), 2)
        avg_conf = round(sum(vals["conf"]) / len(vals["conf"]), 2)
        avg_lat = round(sum(vals["lat"]) / len(vals["lat"]), 2)
        print(f"{engine:<22} {script:<12} {avg_cer:<10} {avg_wer:<10} {avg_conf:<10} {avg_lat:<8}")
    
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Task 3: OCR Evaluation Suite")
    parser.add_argument("--transformer", action="store_true",
                        help="Also evaluate Surya Transformer OCR")
    parser.add_argument("--gt-dir", default=None,
                        help="Directory containing ground-truth .gt.txt files")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("Task 3: OCR Evaluation Suite")
    print("Comparing: Tesseract (Baseline)" + (" vs Surya (Transformer)" if args.transformer else ""))
    print("=" * 60)
    
    all_results = []
    
    # --- Evaluate Tesseract Baseline ---
    print("\n[1/2] Evaluating TESSERACT BASELINE...")
    tesseract_results = evaluate_engine(
        BUILTIN_TESTS, project_root, "TESSERACT_BASELINE",
        run_fn=lambda path: run_tesseract_on_image(path)
    )
    all_results.extend(tesseract_results)
    
    # --- Evaluate Transformer ---
    if args.transformer:
        if TRANSFORMER_AVAILABLE:
            engine = TransformerOCREngine(languages=["hi", "en"])
            engine_label = engine.backend or "TRANSFORMER"
            print(f"\n[2/2] Evaluating {engine_label}...")
            transformer_results = evaluate_engine(
                BUILTIN_TESTS, project_root, engine_label,
                run_fn=lambda path: run_transformer_on_image(path, engine)
            )
            all_results.extend(transformer_results)
        else:
            print("\n[2/2] SKIPPED: No transformer backend available (install surya-ocr or easyocr)")
    
    # --- Print comparison table ---
    print_comparison_table(all_results)
    
    # --- Save results as JSON ---
    output_path = os.path.join(project_root, "output", "evaluation_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
