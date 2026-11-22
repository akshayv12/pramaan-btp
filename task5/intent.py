"""Small question-type cues for selecting supplied evidence, never new facts."""
import re


def question_kind(query):
    if re.search(r'\bwhy\b|\breasons?\b|क्यों|कारण|\bky[ou]n?\b',query,re.I):return 'reason'
    if re.search(r'\bhow\b|कैसे|किस प्रकार',query,re.I):return 'process'
    if re.search(r'\bwho\b|कौन|किसने',query,re.I):return 'person'
    if re.search(r'\bwhere\b|कहाँ|कहां',query,re.I):return 'place'
    if re.search(r'\bwhen\b|\bdate\b|\byear\b|कब|तारीख|वर्ष',query,re.I):return 'date'
    return 'overview' if re.search(r'tell me|what is|what are|overview|के बारे|क्या है',query,re.I) else 'fact'


def requested_count(query):
    names={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
           'एक':1,'दो':2,'तीन':3,'चार':4,'पाँच':5,'पांच':5,'छह':6,'सात':7,'आठ':8,'नौ':9,'दस':10}
    found=re.search(r'(?<![\w\u0900-\u097f])([1-9]|1[0-9]|20|[१-९]|१[०-९]|२०|'+ '|'.join(names)+r')'
                    r'(?![\w\u0900-\u097f])\s+(?:(?:key|main|short|brief|important|major|मुख्य|प्रमुख)\s+)?'
                    r'(?:points?|facts?|reasons?|examples?|features?|crops?|names?|steps?|बातें|बातों|बिंदु|बिन्दु|तथ्य|कारण|उदाहरण|विशेषताएँ|फसलें|चरण)'
                    r'(?![\w\u0900-\u097f])',query,re.I)
    if not found:return None
    value=found[1].lower()
    return names[value] if value in names else int(value)


def requirements(query):
    kind=question_kind(query)
    focus={'reason':'Explain the initiating cause or purpose. Battle tactics, later outcomes or descriptions alone do not answer why.',
           'process':'Explain the requested process or mechanism, rather than only defining its subject.',
           'person':'Identify the requested person or attributed authorship, preserving uncertainty and tradition.',
           'place':'Identify the requested location using the supplied evidence.',
           'date':'Give a date only when the supplied evidence establishes that date. Do not infer a historical date.',
           'overview':'Define the requested subject and cover substantive available content.',
           'fact':'Answer the specific factual request, preserving its constraints.'}[kind]
    return {'kind':kind,'focus':focus,'requested_count':requested_count(query)}


def focus_score(query,text):
    """A bounded selection preference; the LLM still checks question fulfillment."""
    kind=question_kind(query)
    if kind=='reason':
        explicit=bool(re.search(r'\bbecause\b|due to|reason for|led to|prompted|in order to|in response to|के कारण|क्योंकि|इसलिए',text,re.I))
        causal=bool(re.search(r'\b(?:abduct\w*|kidnap\w*|rescu\w*|refus\w*|prevent\w*|caus\w*|necessitat\w*)\b|अपहरण|बचाने|इनकार',text,re.I))
        purpose=bool(re.search(r'\b(?:fought|launched|introduced|built|developed|went|sent|created)\b.{0,90}\bto\b',text,re.I))
        return min(1.,.5*explicit+.35*causal+.35*purpose)
    if kind=='date':return .5 if re.search(r'\b\d{3,4}\b|[०-९]{3,4}',text) else 0.
    return 0.
