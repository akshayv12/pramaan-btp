"""Small Unicode utilities for Hindi, Roman Hindi and English evidence."""
import re
import unicodedata
from difflib import SequenceMatcher

STOP=set('a an the of for in on is are was were to and or with from as by this that what how why does do it its who about है हैं का की के में को से और क्या कैसे क्यों पर यह एक था थे किस लोग'.split())
NEGATIONS={'not','never','no','cannot','नहीं','अज्ञात','undeciphered'}


def normalize(text):return re.sub(r'\s+',' ',unicodedata.normalize('NFC',text)).strip()
def tokens(text):return set(re.findall(r'[a-z0-9]+|[\u0900-\u097f]+',normalize(text).casefold()))-STOP


QUERY_FILLER=set('you your we our i me my know tell please explain describe give overview introduction which when where can could would information details exists exist'.split())


def match_tokens(text):
    # Strip Latin accents, preserving Hindi vowel signs and original evidence.
    folded=re.sub(r'[\u0300-\u036f]','',unicodedata.normalize('NFKD',normalize(text))).casefold()
    return tokens(folded)


def query_terms(query):return match_tokens(query)-QUERY_FILLER


def word_spans(text):
    """Bounded adjacent-word aliases: Sundarakanda / Sundara Kanda / Sundara-kanda.

    Do not join across sentence boundaries, discard query constraints, or alter
    original evidence. Each alias contains at most three adjacent Latin words.
    """
    folded=re.sub(r'[\u0300-\u036f]','',unicodedata.normalize('NFKD',text)).casefold()
    words=list(re.finditer(r'[a-z]+',folded));result={}
    for start,word in enumerate(words):
        if word.group() in STOP|NEGATIONS|QUERY_FILLER:continue
        parts=[word.group()]
        for end in range(start+1,min(start+3,len(words))):
            gap=folded[words[end-1].end():words[end].start()]
            if '\n' in gap or '\r' in gap or not re.fullmatch(r'[\s\-‐‑–]+',gap) or any(len(p)<3 for p in parts) or len(words[end].group())<3 or words[end].group() in STOP|NEGATIONS|QUERY_FILLER:break
            parts.append(words[end].group())
            combined=''.join(parts)
            if len(combined)>=6:result[combined]=tuple(parts)
    return result


def query_overlap(query,text):
    terms=query_terms(query);words=match_tokens(text)
    if not terms:return 0.0
    evidence_aliases=word_spans(text);query_aliases=word_spans(query)
    def matches(term):
        if term in words or term in evidence_aliases:return True
        if any(term in parts and compound in words for compound,parts in query_aliases.items()):return True
        return term.isascii() and term.isalpha() and len(term)>=6 and any(
            word.isascii() and word.isalpha() and len(word)>=6 and word[:3]==term[:3]
            and abs(len(word)-len(term))<=2 and SequenceMatcher(None,term,word).ratio()>=.90 for word in words)
    return sum(matches(term) for term in terms)/len(terms)

def similarity(a,b):
    aa,bb=tokens(a),tokens(b)
    return len(aa&bb)/len(aa|bb) if aa and bb else 0.0


def duplicate_similarity(a,b):
    a,b=normalize(a).casefold(),normalize(b).casefold()
    if a==b:return 1.0
    if bool(tokens(a)&NEGATIONS)!=bool(tokens(b)&NEGATIONS):return 0.0
    if set(re.findall(r'\d+(?:\.\d+)?',a))!=set(re.findall(r'\d+(?:\.\d+)?',b)):return 0.0
    aa={a[i:i+5] for i in range(max(0,len(a)-4))};bb={b[i:i+5] for i in range(max(0,len(b)-4))}
    return max(similarity(a,b),len(aa&bb)/len(aa|bb) if aa and bb else 0)


def sentences(text):
    """Keep abbreviation/parenthetical context attached to its factual sentence."""
    abbreviations={'lit','dr','mr','mrs','ms','prof','sr','jr','st','vs','ed','eds','vol','pp','fig','no','e.g','i.e'}
    result=[];start=0;depth=0
    for index,char in enumerate(text):
        if char in '([':depth+=1
        elif char in ')]' and depth:depth-=1
        if char not in '.!?।॥' or depth:continue
        if char=='.':
            if index+1<len(text) and not text[index+1].isspace():continue
            previous=re.search(r'([A-Za-z.]+)$',text[start:index])
            token=previous.group(1).casefold() if previous else ''
            if token in abbreviations or len(token)==1 and token.isalpha():continue
            if index and text[index-1].isdigit() and index+1<len(text) and text[index+1].isdigit():continue
        value=normalize(text[start:index+1])
        if value:result.append(value)
        start=index+1
    value=normalize(text[start:])
    if value:result.append(value)
    return result


def detect_language(query):
    if re.search(r'[\u0900-\u097f]',query):return 'hi'
    roman={'kaise','kaisa','kaisi','kyu','kyun','kyon','kya','bhagwan','mante','maante','hain','batao','bataiye','samjhao','chahiye','kaun','kitna'}
    return 'hi' if set(re.findall(r'[a-z]+',query.lower()))&roman else 'en'


def indian_query(query):
    return bool(tokens(query)&{'india','indian','bharat','rama','ram','ramayana','hindi','sanskrit','upi','भारत','भारतीय','राम','हिंदी','रामायण','संस्कृत'})
