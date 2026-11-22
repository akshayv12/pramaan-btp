"""Unicode-aware, concept-based Indian-context features.

Aliases identify the same concept across scripts; they do not assert source
reliability. Matching uses whole tokens, so 'ram' never matches 'programming'.
Resources are deliberately explicit and can be extended without retraining.
"""
from __future__ import annotations
import unicodedata
from functools import lru_cache

def normalize_text(text: str) -> str:
    # Joiners affect conjunct rendering, not the identity of a Hindi word.
    # Removing them avoids splitting equivalent OCR/digital spellings.
    text = unicodedata.normalize("NFC", text).replace("\u200c", "").replace("\u200d", "")
    return " ".join(text.casefold().split())

def unicode_tokens(text: str) -> list[str]:
    words, current = [], []
    for char in normalize_text(text):
        category = unicodedata.category(char)[0]
        if category in {"L", "N"} or (category == "M" and current):
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words

# The keys are canonical concepts, not additional independent keyword hits.
CONTEXT_ALIASES = {
    "entity_feature": {
        "rama": ("ram", "rama", "राम"),
        "sita": ("sita", "seeta", "सीता"),
        "krishna": ("krishna", "krishn", "कृष्ण"),
        "buddha": ("buddha", "buddh", "बुद्ध"),
        "gandhi": ("gandhi", "mahatma gandhi", "गांधी", "गाँधी"),
        "ashoka": ("ashoka", "ashok", "अशोक"),
        "isro": ("isro", "इसरो", "indian space research organisation"),
        "chandrayaan": ("chandrayaan", "chandrayaan 3", "chandrayaan-3", "चंद्रयान", "चन्द्रयान"),
        "aryabhata": ("aryabhata", "aryabhatta", "आर्यभट्ट", "आर्यभट"),
        "akbar": ("akbar", "अकबर"),
        "tagore": ("tagore", "rabindranath tagore", "टैगोर", "रवीन्द्रनाथ टैगोर", "रवींद्रनाथ टैगोर"),
        "nehru": ("nehru", "नेहरू"),
        "ramanujan": ("ramanujan", "srinivasa ramanujan", "रामानुजन"),
        "ambedkar": ("ambedkar", "अम्बेडकर", "आंबेडकर", "अंबेडकर"),
        "satyajit_ray": ("satyajit ray", "सत्यजित रे", "सत्यजीत रे"),
        "phalke": ("dadasaheb phalke", "दादासाहेब फाल्के"),
    },
    "place_feature": {
        "india": ("india", "indian", "bharat", "भारत", "भारतीय"),
        "varanasi": ("varanasi", "banaras", "kashi", "वाराणसी", "बनारस", "काशी"),
        "ayodhya": ("ayodhya", "अयोध्या"),
        "kerala": ("kerala", "केरल"),
        "delhi": ("delhi", "नई दिल्ली", "दिल्ली"),
        "mumbai": ("mumbai", "bombay", "मुंबई", "बम्बई"),
        "ganga": ("ganga", "ganges", "गंगा"),
        "himalaya": ("himalaya", "himalayas", "himalayan", "हिमालय"),
        "prayagraj": ("prayagraj", "allahabad", "प्रयागराज", "इलाहाबाद"),
        "madhya_pradesh": ("madhya pradesh", "मध्य प्रदेश"),
        "rajasthan": ("rajasthan", "राजस्थान"),
        "tamil_nadu": ("tamil nadu", "तमिलनाडु", "तमिल नाडु"),
        "karnataka": ("karnataka", "कर्नाटक"),
        "chennai": ("chennai", "madras", "चेन्नई"),
        "bengaluru": ("bengaluru", "bangalore", "बेंगलुरु", "बंगलौर"),
        "kashmir": ("kashmir", "कश्मीर"),
        "uttarakhand": ("uttarakhand", "उत्तराखंड"),
        "sundarbans": ("sundarbans", "sunderbans", "सुंदरबन"),
        "indus_valley": ("indus valley", "harappan", "harappa", "सिंधु घाटी", "हड़प्पा"),
        "darjeeling": ("darjeeling", "दार्जिलिंग"),
        "ranthambore": ("ranthambore", "रणथंभौर"),
        "maharashtra": ("maharashtra", "महाराष्ट्र"),
        "bengal": ("bengal", "bengali", "बंगाल", "बंगाली"),
        "gujarat": ("gujarat", "गुजरात"),
    },
    "institution_feature": {
        "iit": ("iit", "iits", "indian institute of technology", "आईआईटी"),
        "asi": ("asi", "archaeological survey of india", "भारतीय पुरातत्व सर्वेक्षण"),
        "isro": ("isro", "इसरो", "indian space research organisation"),
        "sahitya_akademi": ("sahitya akademi", "साहित्य अकादमी"),
        "bhu": ("banaras hindu university", "bhu", "काशी हिन्दू विश्वविद्यालय", "काशी हिंदू विश्वविद्यालय"),
        "sangeet_natak": ("sangeet natak akademi", "संगीत नाटक अकादमी"),
        "national_museum": ("national museum", "राष्ट्रीय संग्रहालय"),
        "nehru_memorial": ("nehru memorial", "नेहरू स्मारक"),
        "indian_railways": ("indian railways", "bharatiya railway", "भारतीय रेलवे", "भारतीय रेल"),
        "niti_aayog": ("niti aayog", "नीति आयोग"),
        "iisc": ("iisc", "indian institute of science", "भारतीय विज्ञान संस्थान"),
        "ntca": ("ntca", "national tiger conservation authority", "राष्ट्रीय बाघ संरक्षण प्राधिकरण"),
        "visva_bharati": ("visva bharati", "vishva bharati", "विश्वभारती"),
        "khadi_commission": ("khadi and village industries commission", "खादी और ग्रामोद्योग आयोग"),
    },
    "primary_text_feature": {
        "ramayana": ("ramayana", "ramayan", "रामायण"),
        "mahabharata": ("mahabharata", "mahabharat", "महाभारत"),
        "vedas": ("veda", "vedas", "वेद", "वेदों"),
        "upanishads": ("upanishad", "upanishads", "उपनिषद्", "उपनिषद", "उपनिषदों"),
        "gita": ("gita", "geeta", "bhagavad gita", "bhagavadgita", "गीता", "भगवद्गीता"),
        "puranas": ("purana", "puranas", "पुराण", "पुराणों"),
        "gitanjali": ("gitanjali", "गीतांजलि", "गीताञ्जलि"),
        "constitution": ("indian constitution", "constitution of india", "भारतीय संविधान", "भारत का संविधान"),
        "arthashastra": ("arthashastra", "अर्थशास्त्र"),
        "yoga_sutras": ("yoga sutra", "yoga sutras", "योगसूत्र"),
    },
    "contextual_vocab_feature": {
        "dharma": ("dharma", "dharm", "धर्म"),
        "puja": ("puja", "pooja", "पूजा"),
        "panchayat": ("panchayat", "panchayati raj", "पंचायत", "पंचायती राज"),
        "yatra": ("yatra", "यात्रा"),
        "kumbh_mela": ("kumbh mela", "kumbh", "कुंभ मेला", "कुम्भ मेला", "कुंभ", "कुम्भ"),
        "mela": ("mela", "मेले", "मेला"),
        "ashram": ("ashram", "आश्रम"),
        "guru": ("guru", "गुरु"),
        "mandir": ("mandir", "मंदिर", "मन्दिर"),
        "ayurveda": ("ayurveda", "ayurved", "आयुर्वेद"),
        "yoga": ("yoga", "yog", "योग"),
        "raga": ("raga", "raag", "राग"),
        "tala": ("tala", "taal", "ताल"),
        "bharatanatyam": ("bharatanatyam", "bharatnatyam", "भरतनाट्यम", "भरतनाट्यम्"),
        "khadi": ("khadi", "खादी"),
        "swadeshi": ("swadeshi", "स्वदेशी"),
        "tandoor": ("tandoor", "तंदूर"),
        "banarasi": ("banarasi", "banarasi silk", "बनारसी"),
        "kanjeevaram": ("kanjeevaram", "kanchipuram silk", "कांजीवरम"),
        "pashmina": ("pashmina", "पश्मीना"),
        "johad": ("johad", "johads", "जोहड़"),
        "baori": ("baori", "baoris", "stepwell", "stepwells", "बावड़ी"),
        "bollywood": ("bollywood", "बॉलीवुड"),
        "project_tiger": ("project tiger", "प्रोजेक्ट टाइगर"),
        "monsoon": ("indian monsoon", "भारतीय मानसून"),
    },
}

@lru_cache(maxsize=512)
def _alias_tokens(alias: str) -> tuple[str, ...]:
    return tuple(unicode_tokens(alias))

def _contains(tokens: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    width = len(needle)
    return bool(width) and any(tokens[i:i + width] == needle for i in range(len(tokens) - width + 1))

def matched_concepts(text: str, category: str | None = None) -> set[str]:
    tokens = tuple(unicode_tokens(text))
    groups = [CONTEXT_ALIASES[category]] if category else CONTEXT_ALIASES.values()
    return {
        concept for group in groups for concept, aliases in group.items()
        if any(_contains(tokens, _alias_tokens(alias)) for alias in aliases)
    }

def devanagari_ratio(text: str) -> float:
    letters = [char for char in text if unicodedata.category(char).startswith("L")]
    return sum("\u0900" <= char <= "\u097f" for char in letters) / len(letters) if letters else 0.0

def extract_context_features(query: str, text: str) -> dict[str, float]:
    query_concepts = matched_concepts(query)
    passage_concepts = matched_concepts(text)
    features = {"script_feature": min(1.0, 2 * devanagari_ratio(text))}
    features.update({name: min(1.0, len(matched_concepts(text, name)) / 2)
                     for name in CONTEXT_ALIASES})
    features["query_context_match"] = (
        len(query_concepts & passage_concepts) / len(query_concepts)
        if query_concepts else 0.0
    )
    return features
