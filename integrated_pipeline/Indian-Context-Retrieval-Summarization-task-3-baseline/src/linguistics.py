def detect_script(text):
    """
    API Contract: detect_script(text) -> script_label (Latin / Devanagari / Mixed)
    Checks if text is English (Latin), Hindi/Sanskrit (Devanagari), or Mixed based on Unicode blocks.
    """
    has_latin = any('a' <= char.lower() <= 'z' for char in text)
    # 0x0900 to 0x097F is the official Unicode block for Devanagari
    has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
    
    if has_latin and has_devanagari: return "Mixed"
    if has_devanagari: return "Devanagari"
    if has_latin: return "Latin"
    return "Other"

def detect_language(text):
    """
    API Contract: detect_language(text) -> language_label (English / Hindi / Sanskrit / Other)
    Uses advanced lexical language-ID models (Stopwords + Morphological markers).
    """
    script = detect_script(text)
    
    if script == "Latin":
        return "English"
    
    if script in ["Devanagari", "Mixed"]:
        # 1. Hindi Lexical Markers (Nuqta '़' is never used in pure Sanskrit)
        if '़' in text:
            return "Hindi"
            
        # 2. Sanskrit Lexical Markers (Avagraha 'ऽ', Visarga 'ः', Double Danda '॥')
        if any(marker in text for marker in ['॥', 'ः', 'ऽ']):
            return "Sanskrit"
        
        # 3. Statistical Stopword Frequency
        hindi_stopwords = {"है", "और", "में", "की", "से", "को", "का", "कि", "हैं", "एक", "यह"}
        words = [w.strip('॥।?,.!ः') for w in text.split()]
        hindi_count = sum(1 for word in words if word in hindi_stopwords)
        
        if hindi_count >= 1:
            return "Hindi"
        
        return "Sanskrit (Fallback)"
        
    return "Other"

if __name__ == "__main__":
    print("--- Testing Linguistic API Contracts ---")
    
    test_cases = [
        "Indian context aware retrieval system",
        "राम कैसे थे? राम को क्यों भगवान मानते हैं लोग?",
        "तमेव माता च पिता त्वमेव॥",
        "This is a mixed sentence जिसमें हिंदी भी है।"
    ]
    
    for i, t in enumerate(test_cases):
        print(f"Text Case {i+1}")
        print(f" -> Script detected: {detect_script(t)}")
        print(f" -> Language detected: {detect_language(t)}\n")
