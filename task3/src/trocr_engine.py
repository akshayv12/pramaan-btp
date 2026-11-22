import os
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image, ImageDraw, ImageFont
    import torch
except ImportError:
    print("Dependencies not yet installed. Please wait for 'pip install transformers torch pillow' to finish.")
    exit(1)

class Phase2_TrOCREngine:
    def __init__(self, model_id="microsoft/trocr-base-printed"):
        """
        Initializes the Encoder-Decoder Architecture.
        Note: We use Microsoft's baseline here for architecture testing. 
        For final Devanagari production, we will swap this ID to an Indic-fine-tuned model.
        """
        print(f"Loading Phase 2 TrOCR Engine: {model_id}...")
        print("This might take a minute to download the weights on first run...")
        
        # 1. The Processor handles the Vision Transformer (ViT) image patching (16x16)
        self.processor = TrOCRProcessor.from_pretrained(model_id, use_fast=False)
        
        # 2. The Model handles the RoBERTa language decoding (Semantic spacebar prediction)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        
        # 3. Hardware optimization (Uses NVIDIA GPU if you have one, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"TrOCR loaded successfully on {self.device}!\n")

    def extract_text_from_line(self, image_path):
        """
        Takes a cropped image of a single line of text and predicts the characters.
        """
        image = Image.open(image_path).convert("RGB")
        
        # Step A: ViT Encoder (Extract Visual Features)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        
        # Step B: RoBERTa Decoder (Autoregressively predict text with grammar context)
        # This completely bypasses the physical "pixel-gap" limitation of Tesseract
        generated_ids = self.model.generate(pixel_values)
        
        # Step C: Convert mathematical tokens back into a human-readable string
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text

# ==========================================
# TEST SCRIPT
# ==========================================
if __name__ == "__main__":
    print("--- Initializing Phase 2 TrOCR Pipeline ---")
    
    # 1. Create a quick dummy image with some text to test the model
    dummy_image_path = "dummy_test_line.jpg"
    img = Image.new('RGB', (400, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Drawing simple text. In reality, OpenCV will crop this line from your PDF.
    d.text((20, 40), "PHASE 2 NEURAL OCR", fill=(0, 0, 0))
    img.save(dummy_image_path)
    
    # 2. Run the Engine
    try:
        engine = Phase2_TrOCREngine()
        print(f"Processing cropped image line: {dummy_image_path}")
        extracted_text = engine.extract_text_from_line(dummy_image_path)
        
        print("\n" + "="*40)
        print("✅ SUCCESS! TrOCR Extracted Text:")
        print(f"--> '{extracted_text}'")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Clean up dummy image
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)
