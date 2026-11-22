"""Resolve user answer-language instructions separately from evidence language."""
import re
from .text import detect_language, normalize

NAMES = {'hindi': 'hi', 'हिंदी': 'hi', 'हिन्दी': 'hi',
         'english': 'en', 'अंग्रेजी': 'en', 'अंग्रेज़ी': 'en', 'अंग्रेज़ी': 'en'}
NAME = '|'.join(NAMES)
DIRECTIVE = re.compile(r'\b(?:answer|reply|respond|write|explain|describe|summari[sz]e|translate|tell|give|say|narrate)\b|'
                       r'बताओ|बताइए|समझाओ|समझाइए|लिखो|लिखिए|उत्तर|जवाब|दीजिए|बता[इई]ये', re.I)


def explicit_answer_language(query):
    """Recognize supported English/Hindi instructions, not a topic's name.

    This reads only the user's query. Source passages continue to use script
    detection. The last explicit instruction wins; a selector is the fallback.
    """
    text = normalize(query).casefold()
    # Quoted phrases can be the subject of a question, rather than instructions.
    text = re.sub(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|(?<!\w)\x27[^\x27\n]*\x27(?!\w)', '', text)
    found = []
    phrases = re.compile(r'\b(?:in|into)\s+(?:simple\s+|standard\s+)?(?P<latin>hindi|english)\b|'
                         r'(?<![\w\u0900-\u097f])(?P<indian>' + NAME + r')\s+(?:में|मे|mein|me)(?![\w\u0900-\u097f])')
    for match in phrases.finditer(text):
        before = re.split(r'[.!?;।]', text[:match.start()])[-1]
        after = text[match.end():]
        if re.search(r"\b(?:not|never|don't)\s+(?:(?:answer|reply|respond|write|explain)\s+)?$", before):
            continue
        if re.match(r'\s*(?:नहीं|नही|मत)(?![\w\u0900-\u097f])', after):
            continue
        if DIRECTIVE.search(before) or DIRECTIVE.search(after) or not before.strip(' ,:'):
            found.append((match.start(), NAMES[match['latin'] or match['indian']]))
    assignment = re.compile(r'\b(?:answer|response|output)\s+language\s*[:=]\s*(' + NAME + r')(?![\w\u0900-\u097f])')
    for match in assignment.finditer(text):
        found.append((match.start(), NAMES[match[1]]))
    return max(found)[1] if found else None


def resolve_answer_language(query, selected=None):
    if selected not in {None, 'hi', 'en'}:
        raise ValueError('target_language must be hi or en')
    explicit = explicit_answer_language(query)
    return {'selected_language': selected, 'query_requested_language': explicit,
            'resolved_language': explicit or selected or detect_language(query),
            'basis': 'explicit_query_instruction' if explicit else 'language_selector' if selected else 'query_language_detection'}


def matches_answer_language(text, language):
    letters=[c for c in text if c.isalpha()]
    devanagari=sum('\u0900'<=c<='\u097f' for c in letters)
    return bool(letters) and (devanagari/len(letters)>=.60 if language=='hi' else devanagari==0)
