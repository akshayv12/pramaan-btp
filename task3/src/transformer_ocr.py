"""
Task 3: Transformer-based OCR Engine
======================================
Fulfills BTP Requirement: "Evaluate freely available transformer-based OCR 
options for English and Devanagari; keep a practical non-transformer OCR 
baseline for comparison."

Primary Engine: Surya OCR (Vision Transformer, requires Docker)
Fallback Engine: EasyOCR (ResNet + BiLSTM + Attention, runs locally)
"""
import os
import re
from PIL import Image

# --- Try Surya first (requires Docker) ---
try:
    from surya.recognition import RecognitionPredictor
    SURYA_AVAILABLE = True
except ImportError:
    SURYA_AVAILABLE = False

# --- Fallback: EasyOCR (pure PyTorch, no Docker) ---
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


def _strip_html_tags(html_text):
    """Strips HTML tags from Surya's block output to get plain text."""
    clean = re.sub(r'<[^>]+>', ' ', html_text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


class TransformerOCREngine:
    """
    Deep Learning OCR engine with automatic backend selection:
      1. Surya (Vision Transformer) -- if Docker is available
      2. EasyOCR (ResNet + BiLSTM + Attention) -- local fallback
    
    Both support Hindi and English natively.
    """
    def __init__(self, languages=None):
        self.languages = languages or ["hi", "en"]
        self.backend = None
        self._surya_rec = None
        self._easyocr_reader = None
        
        # Try Surya first
        if SURYA_AVAILABLE:
            try:
                print("Loading Surya Transformer OCR (Vision Transformer)...")
                self._surya_rec = RecognitionPredictor()
                self.backend = "SURYA_TRANSFORMER"
                print("[SUCCESS] Surya OCR loaded successfully!\n")
            except Exception as e:
                print(f"[WARNING] Surya failed to load: {e}")
                print("Falling back to EasyOCR...\n")
        
        # Fallback to EasyOCR
        if self.backend is None and EASYOCR_AVAILABLE:
            print("Loading EasyOCR (ResNet + BiLSTM + Attention)...")
            print(f"Languages: {self.languages}")
            self._easyocr_reader = easyocr.Reader(
                self.languages,
                gpu=False,
                verbose=False
            )
            self.backend = "EASYOCR_TRANSFORMER"
            print("[SUCCESS] EasyOCR loaded successfully!\n")
        
        if self.backend is None:
            print("[ERROR] No transformer OCR backend available.")
            print("Install one: pip install surya-ocr  OR  pip install easyocr")

    def extract_document(self, image_path):
        """
        Runs transformer OCR on a single image.
        
        Returns: (extracted_text, avg_confidence)
            extracted_text: str
            avg_confidence: float (0.0 to 1.0)
        """
        if self.backend == "SURYA_TRANSFORMER":
            return self._extract_surya(image_path)
        elif self.backend == "EASYOCR_TRANSFORMER":
            return self._extract_easyocr(image_path)
        else:
            return "", 0.0
    
    def _extract_surya(self, image_path):
        """Surya full-page OCR."""
        image = Image.open(image_path).convert("RGB")
        results = self._surya_rec([image])
        page = results[0]
        
        lines = []
        confs = []
        for block in page.blocks:
            if block.skipped or block.error:
                continue
            text = _strip_html_tags(block.html)
            if text:
                lines.append(text)
                confs.append(block.confidence)
        
        final_text = "\n".join(lines)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return final_text, avg_conf
    
    def _extract_easyocr(self, image_path):
        """EasyOCR line-level OCR."""
        results = self._easyocr_reader.readtext(image_path, detail=1, paragraph=False)
        
        lines = []
        confs = []
        for (bbox, text, confidence) in results:
            text = text.strip()
            if text:
                lines.append(text)
                confs.append(confidence)
        
        final_text = "\n".join(lines)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return final_text, avg_conf


# ==========================================
# TEST SCRIPT
# ==========================================
if __name__ == "__main__":
    print("--- Testing Transformer-Based OCR ---")
    
    if not SURYA_AVAILABLE and not EASYOCR_AVAILABLE:
        print("No OCR backend available. Install: pip install easyocr")
    else:
        test_images = [
            ("English", "data/images/english/eng1.jpeg"),
            ("Hindi", "data/images/hindi/hin1.jpeg"),
            ("Sanskrit", "data/images/sanskrit/san1.jpeg"),
        ]
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        engine = TransformerOCREngine(languages=["hi", "en"])
        print(f"Active backend: {engine.backend}\n")
        
        for label, rel_path in test_images:
            full_path = os.path.join(project_root, rel_path)
            if not os.path.exists(full_path):
                print(f"  [SKIP] {label}: {rel_path} not found")
                continue
            
            print(f"--- {label}: {rel_path} ---")
            text, conf = engine.extract_document(full_path)
            print(f"  Text: {text[:150]}")
            print(f"  Confidence: {conf:.2%}")
            print()
