"""
Task 3: Linguistic Analysis & Language Discrimination
=====================================================
Multi-feature weighted token scoring for Devanagari languages
(Hindi, Sanskrit, Marathi, Nepali) with positive morphological validation.
"""

import unicodedata
import re

HI_WORDS = {
    'है', 'हैं', 'था', 'थी', 'थे', 'और', 'में', 'की', 'का', 'के', 'को', 'से',
    'ने', 'पर', 'कि', 'एक', 'यह', 'वह', 'नहीं', 'लिए', 'रहा', 'रहे', 'रही',
    'सकता', 'सकते', 'सकती', 'भी', 'तो', 'ही', 'आप', 'हम', 'मैं', 'अपने', 'अपनी',
    'अपना', 'हुआ', 'हुए', 'हुई', 'कर', 'करना', 'करने', 'किया', 'कहा', 'कहे',
    'जा', 'जाता', 'गया', 'गए', 'गई', 'इस', 'उस', 'इन', 'उन', 'जय', 'श्री',
    'जो', 'सब', 'बहुत', 'कुछ', 'जब', 'तब', 'यदि', 'होता', 'होते', 'होती'
}

SA_WORDS = {
    'च', 'तु', 'हि', 'एव', 'अपि', 'इति', 'तत्', 'यत्', 'सः', 'तस्य', 'अस्ति',
    'सन्ति', 'कः', 'किम्', 'तथा', 'अथ', 'यथा', 'एवम्', 'उवाच', 'तदा', 'कदा',
    'सर्व', 'पुनः', 'नमः', 'यतः', 'ततः', 'अहम्', 'त्वम्', 'वयम्', 'मा', 'कुरु',
    'भवति', 'उक्तम्', 'तथापि', 'आसीत्', 'न', 'वा'
}

SA_SUFFIXES = ('ः', 'म्', 'स्य', 'ेन', 'न्ति', 'ति', 'ाः', 'ानि', 'ायाः', 'ायै', 'ेभ्यः', 'ेषु')
MR_WORDS = {'आहे', 'आहेत', 'नाही', 'आणि', 'मध्ये', 'च्या', 'त्या', 'केले', 'झाले', 'होते'}
NE_WORDS = {'छ', 'छन्', 'हो', 'र', 'मा', 'लाई', 'बाट', 'हुन्छ', 'यो', 'पनि', 'भने', 'गरेको'}


def detect_script(text):
    """
    API Contract: detect_script(text) -> script_label (Latin / Devanagari / Mixed / Other)
    Checks if text is English (Latin), Hindi/Sanskrit (Devanagari), or Mixed based on Unicode blocks.
    """
    if not text:
        return "Other"
    has_latin = any('a' <= char.lower() <= 'z' for char in text)
    # 0x0900 to 0x097F is the official Unicode block for Devanagari
    has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
    
    if has_latin and has_devanagari:
        return "Mixed"
    if has_devanagari:
        return "Devanagari"
    if has_latin:
        return "Latin"
    return "Other"


def extract_devanagari_tokens(text):
    """Normalizes text and extracts clean Devanagari tokens."""
    clean = unicodedata.normalize('NFC', text).replace('\u200c', '').replace('\u200d', '')
    clean = re.sub(r'[।॥0-9\.,!?;:\"\'\(\)\[\]\{\}\-\–\—/\\]', ' ', clean)
    return [t for t in clean.split() if re.search(r'[\u0900-\u097F]', t)]


def detect_language(text):
    """
    API Contract: detect_language(text) -> language_label (English / Hindi / Sanskrit / Marathi / Nepali / Unknown)
    Uses multi-feature weighted token scoring with positive validation, morphological markers,
    and length thresholds.
    """
    if not text or not text.strip():
        return "Unknown"
        
    script = detect_script(text)
    
    if script == "Latin":
        return "English"
        
    toks = extract_devanagari_tokens(text)
    if not toks:
        if script == "Mixed":
            return "English"
        return "Unknown"
        
    n = len(toks)
    s = {'Hindi': 0.0, 'Sanskrit': 0.0, 'Marathi': 0.0, 'Nepali': 0.0}
    
    hi_matches = sum(1 for t in toks if t in HI_WORDS)
    sa_matches = sum(1 for t in toks if t in SA_WORDS)
    sa_suffixes = sum(1 for t in toks if t.endswith(SA_SUFFIXES))
    mr_matches = sum(1 for t in toks if t in MR_WORDS)
    ne_matches = sum(1 for t in toks if t in NE_WORDS)
    
    # 1. Hindi feature scoring
    s['Hindi'] += 6.0 * (hi_matches / n)
    if '़' in text: s['Hindi'] += 1.0
    if any(c in text for c in ['ॅ', 'ऑ']): s['Hindi'] += 1.0
    
    # 2. Sanskrit feature scoring (positive function words + morphology)
    s['Sanskrit'] += 6.0 * (sa_matches / n) + 3.0 * (sa_suffixes / n)
    if 'ऽ' in text: s['Sanskrit'] += 1.0
    if '॥' in text and (sa_matches > 0 or sa_suffixes > 0): s['Sanskrit'] += 0.5
    if re.search(r'[\u0951-\u0954\u1CD0-\u1CFF]', text): s['Sanskrit'] += 2.0
    
    # 3. Marathi and Nepali feature scoring
    s['Marathi'] += 6.0 * (mr_matches / n) + (1.5 if 'ळ' in text else 0)
    s['Nepali']  += 6.0 * (ne_matches / n)
    
    best = max(s, key=s.get)
    
    # Short text guard: don't guess if token count < 5 and evidence is weak
    if n < 5 and s[best] < 1.0:
        return "Unknown"
        
    if s[best] >= 0.8:
        return best
        
    return "Unknown"


if __name__ == "__main__":
    print("--- Testing Linguistic API Contracts ---")
    
    test_cases = [
        "Indian context aware retrieval system",
        "राम कैसे थे? राम को क्यों भगवान मानते हैं लोग?",
        "अतः हम कह सकते हैं कि यह घटना बहुत दुःखद थी।",
        "जय हनुमान ज्ञान गुन सागर। जय कपीस तिहुँ लोक उजागर॥",
        "रामायण बालकाण्ड",
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥",
        "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन॥ इसका अर्थ यह है कि तुम्हें केवल कर्म करने का अधिकार है।"
    ]
    
    for i, t in enumerate(test_cases):
        print(f"Test Case {i+1}: {t[:40]}...")
        print(f" -> Script detected:   {detect_script(t)}")
        print(f" -> Language detected: {detect_language(t)}\n")
