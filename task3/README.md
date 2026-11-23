# Task 3: English & Devanagari OCR & Text Preparation

> **Pramaan:** Indian-Context-Aware Multilingual and Multimodal Web Retrieval & Summarization  
> **Team:** Praveen Kumar (230041018) & Vivek Tej (230041014)  
> **Supervisors:** Dr. Puneet Gupta & Dr. Vijay Kumar Sohani  
> **Department:** Department of Electrical Engineering, Indian Institute of Technology Indore  

---

## 📖 Table of Contents
1. [Overview & Role in Pramaan](#-overview--role-in-pramaan)
2. [Problem Statement & Scope](#-problem-statement--scope)
3. [Key Architectural Innovations](#-key-architectural-innovations)
4. [Tech Stack & Dependencies](#-tech-stack--dependencies)
5. [Data Contract & Schema Specifications](#-data-contract--schema-specifications)
6. [Benchmarking & Empirical Results](#-benchmarking--empirical-results)
7. [Drawbacks, Known Limitations & Future Work](#-drawbacks-known-limitations--future-work)
8. [Installation & Setup Instructions](#-installation--setup-instructions)
9. [Execution Guide](#-execution-guide)
   - [A. Running End-to-End Until Task 3 (Combined Tasks 1, 2, and 3)](#a-running-end-to-end-until-task-3-tasks-1-2-and-3)
   - [B. Running Task 3 Alone (Independent Module Execution)](#b-running-task-3-alone-independent-module-execution)
10. [Directory Structure & Code Map](#-directory-structure--code-map)

---

## 🌟 Overview & Role in Pramaan

**Pramaan** is an end-to-end question-answering and information retrieval platform designed specifically for the cultural, historical, and linguistic nuances of the Indian context. 

The overall platform is divided into five specialized, decoupled modules connected via strict JSON data contracts:
```text
┌──────────────┐     ┌──────────────┐     ┌────────────────────────────────┐
│    Task 1    │     │    Task 2    │     │             TASK 3             │
│ Multilingual │ ──> │  Web Search  │ ──> │ English & Devanagari OCR,      │
│ Query Expand │     │  & Scraping  │     │ Extraction & Text Preparation  │
└──────────────┘     └──────────────┘     └────────────────────────────────┘
                                                           │
                                                           ▼
┌──────────────┐                          ┌────────────────────────────────┐
│    Task 5    │                          │             Task 4             │
│ Multi-Doc    │ <─────────────────────── │     Indian-Context Semantic    │
│ Summarize    │                          │      Evidence Prioritization   │
└──────────────┘                          └────────────────────────────────┘
```

### What is Task 3's Specific Role?
Task 2 dynamically discovers and downloads heterogeneous, unvetted web assets (raw HTML pages, digital PDFs, scanned PDF documents, smartphone photos of museum boards, temple inscriptions, and book scans).

**Task 3 acts as the foundational content ingestion and normalization layer.** It converts these noisy, multi-format binary and web assets into clean, passage-chunked, machine-readable text in **English**, **Hindi**, and **Sanskrit**. It tags each passage with provenance (page numbers, bounding boxes, file references), language/script metadata, and calibrated OCR confidence scores so downstream retrieval (Task 4) and synthesis (Task 5) can trace and verify every claim.

---

## 🎯 Problem Statement & Scope

### The Core Challenges:
1. **Multimodal Format Heterogeneity:** Web retrievals range from clean UTF-8 HTML to scanned dual-column archival PDFs and compressed mobile camera captures (`.jpg`, `.png`, `.webp`).
2. **Devanagari Complex Script Orthography:** Unlike Latin text, Devanagari features complex two-dimensional conjunct characters (*samyuktaksaras*), modified base consonants (*halanta/virama*), and dependent vowel signs (*matras*) that hang above, below, or across the continuous top headline (*shirorekha*). Classical OCR engines (e.g., Tesseract) frequently fragment these ligatures, producing catastrophic character error rates.
3. **Multi-Script & Multi-Language Disambiguation:** Hindi, Sanskrit, Marathi, and Nepali all share the Devanagari script. Identifying the script via Unicode block ranges alone is insufficient to distinguish Hindi from Sanskrit.
4. **Computational Latency vs. Accuracy Tradeoff:** Running heavy Vision Transformer neural networks across hundreds of web documents on the fly is prohibitive for live search, while relying solely on lightweight legacy OCR fails on Indic scripts.
5. **Near-Duplicate Pollution:** Search crawlers inevitably retrieve mirrored articles, blog syndications, and identical excerpts.

### Task 3 Deliverables:
- **`extract_and_prepare(document_path)` API:** Universal entry point for any incoming document.
- **Smart 3-Lane Routing:** Bypasses OCR for digital text; routes Latin/clean scans to fast baseline OCR; routes Devanagari/Indic scans to deep Vision Transformer models.
- **Dynamic Auto-Escalation:** Monitors Tesseract confidence; automatically escalates low-confidence scans (<60%) to Transformer OCR.
- **MinHash LSH Deduplication:** Eliminates identical and near-duplicate passages ($	ext{Jaccard} \ge 0.85$).
- **Multi-Feature Language ID:** Distinguishes Sanskrit from Hindi and other Devanagari-script languages.
- **Strict Data Contract Handoff:** Outputs standardized `task3_output.json` ready for Task 4 semantic vector indexing.

---

## ⚡ Key Architectural Innovations

```text
Incoming Document (PDF / Image / HTML)
                │
                ▼
   [ Text-Layer Detection Router ]
       │                  │
       │ (Has Text Layer) │ (Scanned / Image)
       ▼                  ▼
[ Lane 1: Direct ]   [ Image Preprocessing ]
  PyMuPDF / BS4       (Grayscale, CLAHE, Otsu, Deskew)
  (<40ms latency)         │
                          ├─────────────────────────┐
                          ▼                         ▼
                  [ Lane 2: Fast CPU ]      [ Lane 3: Vision Transformer ]
                  Tesseract Baseline        Surya ViT / EasyOCR Fallback
                  (Latin / English)         (Devanagari / Indic Priority)
                          │                         │
                          │ (Confidence < 60%?)     │
                          └────> [ AUTO-ESCALATE ] ─┘
                                        │
                                        ▼
                          [ Unicode Normalization (NFC) ]
                          [ Script & Language Detection ]
                          [ Semantic Passage Chunking ]
                          [ MinHash LSH Deduplication ]
                                        │
                                        ▼
                           task3_output.json (To Task 4)
```

### 1. Smart 3-Lane Routing & Dynamic Auto-Escalation
- **Lane 1 (Direct Extraction):** Uses PyMuPDF (`fitz`) and `BeautifulSoup4` to extract digital text directly. Bypasses OCR for **54.5% of PDFs** and all HTML files in $<40\,	ext{ms}$ with 100% confidence.
- **Lane 2 (Tesseract Baseline):** Leverages Tesseract 5.x with `hin+eng` trained data for fast CPU-bound OCR on English documents and clean, simple scans.
- **Lane 3 (Vision Transformer):** Routes Devanagari, mixed-script, and degraded images to **Surya OCR** (Vision Transformer encoder-decoder) with local **EasyOCR** fallback.
- **Dynamic Auto-Escalation:** If Lane 2 produces an OCR confidence score below $60.0\%$ (flagged as `LOW` reliability), the pipeline automatically re-routes the page through Lane 3 to rescue degraded text.

### 2. Indic-Optimized Image Preprocessing
Before passing noisy images to OCR models, Task 3 executes a 3-stage OpenCV enhancement pipeline:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Corrects non-uniform shadows and lighting on mobile camera photos without over-amplifying background noise (`clipLimit=2.0`, `tileGridSize=(8,8)`).
- **Adaptive Otsu Binarization:** Computes optimal global thresholding between ink and paper background.
- **Affine Deskew Correction:** Uses Hough transform line detection and minimum-area bounding boxes to correct rotational skew within $\pm 45^\circ$.

### 3. Unicode Normalization & Shirorekha Continuity
- Enforces Unicode **NFC (Canonical Decomposition, followed by Canonical Composition)** normalization (`unicodedata.normalize('NFC', ...)`).
- Canonicalizes *nukta* characters and compound *matras* into single standardized Unicode codepoints.
- Repairs broken hyphenations, split akshara fragments, and removes OCR artifacts while preserving whitespace and paragraph flow.

### 4. Dual-Stage Script & Language Disambiguation
- **Script Identification (`detect_script`):** Performs character frequency analysis across Unicode blocks:
  - Latin: `U+0041` to `U+007A`
  - Devanagari: `U+0900` to `U+097F`
  - Labels: `Latin`, `Devanagari`, `Mixed`, or `Other`.
- **Language Identification (`detect_language`):** Employs a multi-feature scoring model that separates Hindi from Sanskrit using:
  - Morphological indicators (Sanskrit visarga `: / ः`, halanta terminal clusters, grammatical endings `-स्य`, `-म्`, `-तस्`).
  - Stopword frequency dictionaries (Hindi: `है`, `में`, `और`, `था`, `कि`; Sanskrit: `अस्ति`, `च`, `इति`, `एव`, `तथा`).
  - Secondary fallback to Marathi and Nepali detection.

### 5. Semantic Passage Chunking & MinHash LSH Deduplication
- **Passage Chunker:** Breaks large document bodies into semantic passages of 150–250 words with a 30-word sliding window overlap, preserving sentence boundaries for Task 4 dense vector embeddings.
- **MinHash LSH Deduplicator:** Computes 128 permutation hash signatures using `datasketch`. Detects and prunes overlapping chunks with a Jaccard similarity index $\ge 0.85$, eliminating syndicated news mirrors and search engine duplicates.

---

## 🛠 Tech Stack & Dependencies

| Layer | Technology / Library | Purpose |
|---|---|---|
| **Programming Language** | Python 3.12 | Primary execution runtime |
| **Deep Learning OCR** | Surya OCR (`surya-ocr`) | Vision Transformer encoder-decoder for high-fidelity Devanagari & English |
| **Fallback Neural OCR** | EasyOCR (`easyocr`) | PyTorch-native ResNet + BiLSTM + Attention fallback |
| **Classical OCR Baseline** | Tesseract 5.x (`pytesseract`) | Fast CPU baseline with `eng` and `hin` trained data |
| **PDF Processing** | PyMuPDF (`fitz`), `pdfplumber` | Text-layer detection, digital extraction, and raster rendering |
| **HTML Scraping** | `beautifulsoup4`, `lxml` | HTML document text extraction and tag cleaning |
| **Computer Vision** | OpenCV (`cv2`), `Pillow` (`PIL`) | CLAHE contrast enhancement, Otsu binarization, affine deskew |
| **Text Normalization** | Python `unicodedata` | NFC normalization, Devanagari matra and nukta composition |
| **Linguistics & LangID** | `langdetect`, Custom Lexical Scorer | Unicode code-point analysis and Hindi vs. Sanskrit disambiguation |
| **Deduplication** | `datasketch` | MinHash Locality-Sensitive Hashing (LSH) for near-duplicate pruning |
| **Evaluation Metrics** | `jiwer` | Character Error Rate (CER) and Word Error Rate (WER) computation |

---

## 📋 Data Contract & Schema Specifications

Task 3 adheres strictly to the decoupled pipeline schema:

### Input Contract: `task3_input.json` (from Task 2)
Task 2 delivers a JSON manifest specifying collected documents, their local storage paths, and download status:
```json
{
  "task3_input": {
    "schema_version": "1.0",
    "metadata": {
      "query": "राम कैसे थे? राम को क्यों भगवान मानते हैं लोग?",
      "total_documents": 35
    },
    "documents": [
      {
        "doc_id": "doc_001",
        "url": "https://example.org/article.html",
        "saved_path": "downloads/doc_001.pdf",
        "content_type": "pdf",
        "needs_ocr": true,
        "extraction_status": "success",
        "robots_allowed": true,
        "licence_permits_processing": true
      }
    ],
    "processing_requirements": {
      "ocr_required": ["doc_001"],
      "text_extracted": []
    }
  }
}
```

### Output Contract: `task3_output.json` (Handoff to Task 4)
Task 3 writes the enriched, passage-chunked output consumed by Task 4:
```json
{
  "task3_output": {
    "schema_version": "1.0",
    "metadata": { ... },
    "summary": {
      "processed": 33,
      "duplicate": 2,
      "empty": 0,
      "skipped": 0,
      "failed": 0,
      "total_documents": 35,
      "total_passages": 57
    },
    "documents": [
      {
        "document_id": "doc_001",
        "extraction_method": "OCR",
        "ocr_engine": "SURYA_VISION_TRANSFORMER",
        "dominant_script": "Devanagari",
        "dominant_language": "Hindi",
        "total_passages": 3,
        "duplicates_removed": 0,
        "total_words": 542,
        "avg_ocr_confidence": 95.6,
        "reliability": "HIGH",
        "passages": [
          {
            "passage_id": "doc_001_p001",
            "page_number": 1,
            "detected_script": "Devanagari",
            "detected_language": "Hindi",
            "ocr_confidence": 95.6,
            "reliability": "HIGH",
            "text": "भारतीय परंपरा में श्रीराम को मर्यादा पुरुषोत्तम के रूप में पूजा जाता है...",
            "word_count": 182
          }
        ]
      }
    ]
  }
}
```

---

## 📊 Benchmarking & Empirical Results

Our evaluation suite (`src/evaluate.py`) rigorously tested both engines against a diverse ground-truth corpus containing dense English prints, degraded mobile camera captures, scanned Hindi literary books, and classical Sanskrit stotrams.

### 1. Comparative Performance (Tesseract Baseline vs. Surya ViT)

| Language / Script Cohort | Engine | Character Error Rate (CER %) | Word Error Rate (WER %) | Mean Confidence (%) | Mean Latency (s) |
|---|---|---|---|---|---|
| **English Cohort** | Tesseract 5.x Baseline | 21.56% | 31.41% | 71.4% | 0.18 s |
| **English Cohort** | **Surya Vision Transformer** | **1.31%** | **6.43%** | **95.5%** | 1.06 s |
| *Improvement* | *Relative Reduction* | **93.9% reduction** | **79.5% reduction** | *+24.1% gain* | — |
| **Devanagari Cohort** | Tesseract 5.x Baseline | 62.99% | 83.24% | 36.2% | 0.25 s |
| **Devanagari Cohort** | **Surya Vision Transformer** | **3.06%** | **15.03%** | **93.9%** | 1.66 s |
| *Improvement* | *Relative Reduction* | **95.1% reduction** | **81.9% reduction** | *+57.7% gain* | — |

$$\Delta	ext{CER}_{	ext{Dev}} = rac{62.99\% - 3.06\%}{62.99\%} 	imes 100\% = 95.14\% pprox \mathbf{95.1\%}$$
$$\Delta	ext{WER}_{	ext{Dev}} = rac{83.24\% - 15.03\%}{83.24\%} 	imes 100\% = 81.94\% pprox \mathbf{81.9\%}$$

### 2. Full 35-Document Real-World Batch Ingestion

Evaluating the end-to-end pipeline on 35 real-world heterogeneous web documents collected by Task 2:
- **Total Ingested Documents:** 35
- **Direct Extraction Bypass Rate:** 54.5% of PDFs (17.1% of entire corpus) bypassed OCR via PyMuPDF in $<40\,	ext{ms}$.
- **Total Words Extracted:** 9,521 words
- **Total Semantic Passages Generated:** 57 passages
- **Near-Duplicate Passages Pruned:** 2 passages removed via MinHash LSH ($	ext{Jaccard} \ge 0.85$)
- **Mean Pipeline Latency:** 1.14 seconds per document
- **Mean Output Confidence:** 83.4%
- **Pipeline Completion Reliability:** **100.0%** (zero unhandled exceptions or crashes across the entire batch)

---

## ⚠️ Drawbacks, Known Limitations & Future Work

While Task 3 achieves state-of-the-art accuracy for Indian web retrieval, several engineering tradeoffs and limitations exist:

1. **Inference Latency on Pure CPU:**
   - Vision Transformer models (Surya ViT) take $\sim 1.66\,	ext{s}$ per page on standard CPU vs. $0.25\,	ext{s}$ for Tesseract. In live on-the-fly search where users wait for answers, processing dozens of scanned pages simultaneously requires GPU acceleration or quantized ONNX models.
2. **Severely Degraded Historical Manuscripts:**
   - While modern printed Devanagari and high-contrast scans achieve $>96\%$ accuracy, degraded palm-leaf manuscripts or low-resolution temple stone inscriptions with eroded characters still suffer from matra confusion and character substitution.
3. **Complex Multi-Column & Magazine Layout Linearization:**
   - Text reading order is sorted geometrically. Highly intricate magazine layouts with interspersed advertisements, floating sidebars, and non-rectangular text blocks can occasionally result in interleaved reading order. Integrating layout-aware models (e.g., LayoutLM or Marker) is planned for Phase 2.
4. **Memory Footprint of Neural OCR:**
   - Loading PyTorch and the Surya transformer weights requires $\sim 1.2\,	ext{GB}$ of RAM. In low-memory serverless environments, Task 3 gracefully defaults to Tesseract or EasyOCR.

---

## 💻 Installation & Setup Instructions

### 1. Prerequisites
- **Python 3.12** installed (recommended for compatibility across all pipeline packages).
- **Git** installed.
- **Tesseract OCR Engine** installed on your operating system:
  - **Windows:** Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).  
    *Important:* During setup, expand **Additional script data** and check **Devanagari script**, and expand **Additional language data** and check **Hindi** (`hin`) and **Sanskrit** (`san`).
  - **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-san libtesseract-dev
    ```
  - **macOS:**
    ```bash
    brew install tesseract tesseract-lang
    ```

### 2. Clone Repository & Setup Environment
From your command prompt / terminal:

```bash
# Clone the repository
git clone https://github.com/akshayv12/pramaan-btp.git
cd pramaan-btp

# Create virtual environment
python -m venv .integration-venv

# Activate virtual environment
# Windows PowerShell:
.\.integration-venv\Scripts\Activate.ps1
# Windows CMD:
.\.integration-venv\Scriptsctivate.bat
# Linux / macOS:
source .integration-venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 3. Install Dependencies
Task 3 provides modular requirements files:

```bash
# Option A: Baseline dependencies (Direct extraction + Tesseract baseline + Normalization)
pip install -r task3/requirements-baseline.txt

# Option B: Full Deep Learning Pipeline (Includes Surya ViT, EasyOCR, and jiwer for evaluation)
pip install -r task3/requirements.txt
```

> **Note for Windows Users:** If Tesseract is installed in the default location (`C:\Program Files\Tesseract-OCR	esseract.exe`), `pytesseract` detects it automatically. If it is in a custom path, ensure `tesseract.exe` is added to your system `PATH`, or set `pytesseract.pytesseract.tesseract_cmd` accordingly.

---

## 🚀 Execution Guide

### A. Running End-to-End Until Task 3 (Tasks 1, 2, and 3)
To run the query expansion (Task 1), web crawling & scraping (Task 2), and document extraction/OCR (Task 3) in sequence:

#### 1. Using the Pipeline Orchestrator (`run_pipeline.py`):
```powershell
# From the repository root (with virtual environment active):
python pipeline/run_pipeline.py --with-task3 --query "राम कैसे थे? राम को क्यों भगवान मानते हैं लोग?" --max-queries 4 --top-k 5
```
**What this does:**
1. **Task 1** generates Hindi and English search query variants $	o$ saves `task1_input.json`.
2. **Task 2** performs on-the-fly web scraping, collects HTML/PDFs/images, checks permissions $	o$ saves `task3_input.json`.
3. **Task 3** automatically ingests `task3_input.json`, routes documents through the 3-Lane pipeline, chunks passages, and outputs `task3_output.json`.

#### 2. Using the Integrated Runner (`run_integrated_pipeline.py`):
```powershell
python pipeline/run_integrated_pipeline.py --query "What is the historical significance of Varanasi?" --output-dir outputs/varanasi_run
```
*(Stops cleanly at `outputs/varanasi_run/task3_output.json` if Task 4 and 5 flags are omitted).*

---

### B. Running Task 3 Alone (Independent Module Execution)

You can run Task 3 completely independently in several modes:

#### Mode 1: Batch Process a Task 2 Handoff Manifest (Production Mode)
Run the 3-Lane batch router against any `task3_input.json` file:
```powershell
# Auto Mode (Lane 1 Direct + Lane 2 Tesseract + Lane 3 Surya on Devanagari + Auto-Escalation)
python task3/src/process_manifest.py task3_input.json --output outputs/task3_output.json --engine auto

# Force Vision Transformer across all OCR-required documents:
python task3/src/process_manifest.py task3_input.json --output outputs/task3_output.json --engine transformer

# Force fast Tesseract baseline only:
python task3/src/process_manifest.py task3_input.json --output outputs/task3_output.json --engine tesseract
```

#### Mode 2: Batch Process a Raw Folder of Images and PDFs
Process any local directory containing mixed documents (e.g., the `New data` folder):
```powershell
# Run batch extraction using Surya Transformer OCR:
python task3/src/main.py --data-dir "task3/New data/New data" --output-dir "task3/outputs_surya" --engine transformer

# Run batch extraction using Tesseract Baseline:
python task3/src/main.py --data-dir "task3/New data/New data" --output-dir "task3/outputs_tesseract" --engine tesseract
```
This generates:
- Individual `<filename>_extracted.json` files for each document.
- `pipeline_summary.json` containing total word counts, passage counts, duplicate counts, and document reliability ratings.

#### Mode 3: Single Document Programmatic API Execution
You can invoke the Task 3 API directly from Python:
```python
from task3.src.main import extract_and_prepare
import json

# Process any PDF, Image, or HTML file:
result = extract_and_prepare(
    file_path="task3/New data/New data/images/hin1.jpg",
    engine_type="transformer"  # or "tesseract"
)

print(f"Document ID:        {result['document_id']}")
print(f"Extraction Method:  {result['extraction_method']}")
print(f"Detected Script:    {result['dominant_script']}")
print(f"Detected Language:  {result['dominant_language']}")
print(f"OCR Confidence:     {result['avg_ocr_confidence']:.2f}% ({result['reliability']})")
print(f"Passages Extracted: {result['total_passages']}")

# Inspect the first passage
if result['passages']:
    print("
Passage 1 Text:")
    print(result['passages'][0]['text'][:300] + "...")
```

#### Mode 4: Run the Formal Evaluation Suite (CER & WER Benchmarks)
To reproduce our empirical benchmark results comparing Tesseract Baseline vs. Surya Transformer OCR:
```powershell
# Benchmark both engines side-by-side on ground-truth corpus:
python task3/src/evaluate.py --transformer

# Benchmark Tesseract baseline only:
python task3/src/evaluate.py
```
This prints the full comparison table with Character Error Rates (CER), Word Error Rates (WER), confidence scores, and per-page latency.

---

## 📂 Directory Structure & Code Map

```text
task3/
├── README.md                      # Comprehensive Task 3 documentation (this file)
├── requirements-baseline.txt      # Lightweight dependencies (PyMuPDF, pytesseract, OpenCV, datasketch)
├── requirements.txt               # Full dependencies (including surya-ocr, easyocr, jiwer)
├── New data/                      # Evaluation corpus of real-world images & PDFs
│   └── New data/
│       ├── images/                # English, Hindi, Sanskrit, and Mixed image scans (.jpg, .png, .webp)
│       └── pdfs/                  # Digital and scanned multi-page PDFs
├── output_tesseract/              # Sample JSON outputs generated from baseline runs
└── src/
    ├── main.py                    # Core pipeline orchestrator & extract_and_prepare() API
    ├── process_manifest.py        # Task 2 -> Task 3 batch manifest router with 3-Lane routing & auto-escalation
    ├── router.py                  # Format inspector & digital text-layer detection (PyMuPDF)
    ├── extractor.py               # Direct text extraction from HTML (BS4) and digital PDFs
    ├── ocr_baseline.py            # Tesseract 5.x integration + CLAHE/Otsu/Deskew preprocessing
    ├── transformer_ocr.py         # Deep Learning Vision Transformer (Surya ViT) + EasyOCR fallback
    ├── linguistics.py             # Unicode script detection & Hindi vs. Sanskrit multi-feature scoring
    ├── normalizer.py              # Indic NFC Unicode normalization, matra/nukta repair & cleanup
    ├── deduplicator.py            # MinHash LSH passage deduplication (datasketch)
    ├── evaluate.py                # Formal benchmarking suite (CER/WER evaluation via jiwer)
    └── evaluate_baseline.py       # Standalone test script for CER/WER baseline verification
```

---

## 🔗 Handoff & Downstream Integration

- **Upstream Connection (from Task 2):** Ingests raw downloads and metadata produced by `task2/main.py`.
- **Downstream Connection (to Task 4):** Writes `task3_output.json` directly consumed by `task4/process_task3_output.py`. Task 4 uses the extracted passages, provenance, and confidence scores to calculate Indian-context relevance and dense vector embeddings.
- **Final Integration (to Task 5):** Passages retained by Task 4 are passed to Task 5 for multi-document synthesis, claim citation grounding, and source reliability scoring.
