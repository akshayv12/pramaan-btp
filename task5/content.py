"""Cheap evidence-content checks; exact quotes alone do not make useful answers."""
import re
from .text import normalize, query_overlap, query_terms, sentences,match_tokens

UI_TEXT=re.compile(r'thank you for (?:your )?(?:feedback|response)|\b(?:start|take|related|ultimate) (?:a |the )?quiz\b|\bselect (?:a match|an? option|the option)\b|\b(?:sign in|log in|subscribe|cookie policy|accept all|reject all|privacy policy|copy (?:link|to clipboard)|view my results|please provide|share (?:on|via)|skip to content|jump to content|this article is about|for other uses|more from|more like this)\b|ignore.*instructions|system prompt|api.key|read more|सामग्री पर जाएँ|आइए.*समझ|in this article we',re.I)
ACCOUNT_TEXT=re.compile(r'\b(?:you are logged in|active subscriptions?|manage subscriptions|subscription benefits|account settings|customer\s*support|member.only benefits|unlock.*stories|commenting platform|comments have to be in English|register and login|please register|users can access their older comments|quiz is your gateway|quiz will challenge|challenge your knowledge)\b',re.I)
QUESTION=re.compile(r'^(?:[\d.)\s]*)(?:who|what|when|where|why|how|which|would you|do you|can you|क्या|कौन|किसने|कब|कहाँ|कैसे)\b',re.I)
BIBLIOGRAPHY=re.compile(r'\b(?:ISBN|ISSN|doi:|retrieved|archived from|accessed on|all rights reserved)\b|\b(?:vol\.?|pp\.?|chapters?)\s*\d|^[\"“].{0,140}\bno\.?\s*\d|\b(?:18|19|20)\d{2}\)\s*:\s*\d+\s*[-–]\s*\d+',re.I)
TRAILING=re.compile(r'\b(?:by|of|the|a|an|and|or|to|with|from|into|through|between|because|that|which)\s*$',re.I)


def concise_statement(quote):
    """Omit only balanced pronunciation annotations, keeping the full source quote.

    Numbers, negation, attribution and factual qualifiers prevent omission. This
    deterministic deletion is checked again by the support assessor.
    """
    spans=[];depth=0;start=None
    for index,char in enumerate(quote):
        if char=='(':
            if depth==0:start=index
            depth+=1
        elif char==')' and depth:
            depth-=1
            if depth==0:
                annotation=quote[start:index+1]
                pronunciation=re.search(r'\b(?:pronounced|pronunciation|phonetic|IAST|IPA)\b',annotation,re.I)
                qualifier=re.search(r'\d|\b(?:not|never|only|unless|except|possibly|perhaps|sometimes|may|according|tradition|claimed|disputed)\b|नहीं',annotation,re.I)
                if pronunciation and not qualifier:spans.append((start,index+1))
    for start,end in reversed(spans):quote=quote[:start]+' '+quote[end:]
    return normalize(quote)


def content_reason(text,require_statement=True):
    if not (20 if require_statement else 8)<=len(text)<=600:return 'length is unsuitable for a concise factual sentence'
    if UI_TEXT.search(text) or ACCOUNT_TEXT.search(text):return 'website controls, promotion or feedback'
    if '?' in text or '？' in text or QUESTION.search(text):return 'question instead of an answer statement'
    if BIBLIOGRAPHY.search(text):return 'bibliographic or page metadata'
    if TRAILING.search(text):return 'unfinished sentence'
    if re.match(r'^[a-z]+\s*;',text):return 'overlapping chunk begins with an unfinished fragment'
    if not re.search(r'[\u0900-\u097f]',text):
        words=re.findall(r'[A-Za-z]+',text)
        if require_statement and len(words)<5:return 'title or short fragment'
        finite=re.search(r'\b(?:am|is|are|was|were|has|have|had|can|may|will|would|could|must|grew|became|wrote|built|said|taught|gave|began|made|led|took|won|lost|sent|brought|spread|help|helps|provide|provides|maintain|maintains|include|includes|contain|contains|narrate|narrates|describe|describes|preserve|preserves|tell|tells|regard|regards|open|opens|allow|allows|offer|offers|cover|covers|collect|collects|grant|grants|keep|keeps|consist|consists|depict|depicts|mention|mentions|exist|exists|increase|increases|decrease|decreases|change|changes|support|supports|permit|permits|give|gives|hold|holds|mean|means|refer|refers|use|uses|call|calls|enable|enables|create|creates|serve|serves|become|becomes|explain|explains|explore|explores|form|forms|remain|remains|follow|follows|represent|represents|develop|develops|differ|differs|reach|reaches|live|lives|work|works)\b',text,re.I) or re.search(r'\b[a-z]{3,}ed\b',text)
        if require_statement and not finite:return 'title or fragment without a declarative predicate'
        if require_statement and not re.search(r'[.!][\"\'”’]?$',text) and len(words)<8:return 'unfinished short fragment'
    return ''


ANTECEDENT_PRONOUN=re.compile(r'\b(?:he|she|her|him|they|them|it)\b',re.I)


def usable_sentences(passage,*,include_antecedent=False):
    html=passage.metadata.get('source_metadata',{}).get('content_type')=='html'
    lines=passage.text.splitlines()
    # Some saved pre-repair HTML extractions put almost every word on a line.
    # Treat that clearly fragmented layout as soft wrapping, retaining exact
    # normalized source spans. Normal HTML paragraphs remain separate blocks.
    word_wrapped=html and len(lines)>=12 and sum(len(line.split())<=2 for line in lines)>=.7*len(lines)
    blocks=lines if html and not word_wrapped else [passage.text]
    for block in blocks:
        previous=''
        for quote in sentences(block):
            # Keep real bracketed content (e.g. phonetic transcriptions). Joining
            # around removed spans would produce an unsupported composite quote.
            text=concise_statement(quote)
            if content_reason(text) or unattributed_belief(passage,text):
                previous='';continue
            original_quote=quote
            if include_antecedent and previous and ANTECEDENT_PRONOUN.search(text):
                # A standalone "her"/"he" can lose the named referent. Keep at
                # most one immediately preceding, usable sentence from this same
                # block. Never bridge a heading, question, control or paragraph.
                contextual_quote=normalize(previous+' '+quote)
                contextual_text=concise_statement(contextual_quote)
                if not content_reason(contextual_text) and not unattributed_belief(passage,contextual_text):
                    text,quote=contextual_text,contextual_quote
            yield text,quote
            previous=original_quote


def unattributed_belief(passage,text):
    religious_context=re.search(r'sampradh?aya|devotional|scriptures?|religious traditions?|divine origin|spiritual lineage',passage.text,re.I)
    emphatic=re.search(r'\b(?:actual|true|real|original)\s+(?:author|creator|god)\b|beyond dispute',text,re.I)
    attributed=re.search(r'according to|traditionally|tradition (?:holds|states)|believ|regard|is said|attributed',text,re.I)
    return bool(religious_context and emphatic and not attributed)


def overview_query(query):
    if re.search(r'नाम|किसके|किसने|किन |कौन|कब|कहाँ|कितने|कितनी',query):return False
    return len(query_terms(query))<=3 and bool(re.search(r'\b(?:what (?:is|are|was|do you know)|tell me|overview|introduction|explain|describe)\b|क्या.*है|के बारे में|batao|bataiye',query,re.I))


def definition_match(query,text):
    if not overview_query(query):return 0.0
    # Prefer a statement about the requested subject over a title that merely
    # contains its name. This is an answer-selection heuristic, not truth proof.
    predicate=re.search(r'\b(?:is|are|was|were|narrates|describes|includes|contains|consists)\b|है|हैं',text,re.I)
    if not predicate:return 0.0
    if predicate.group().lower() in {'was','were'} and not re.match(r'\s+(?:an?|the|one|among)\b',text[predicate.end():],re.I):return 0.0
    subject=re.sub(r'\([^)]*\)',' ',text[:predicate.start()])
    if re.match(r'\s*(?:in|on|during|after|before)\b',subject,re.I):return 0.0
    if not query_overlap(query,subject) or len(subject.split())>12:return 0.0
    return 1.0 if query_overlap(query,' '.join(subject.split()[:4])) else .5


def sentence_match(query,text,passage,*,allow_ranked=False):
    if overview_query(query) and not re.search(r'\b(?:disambiguat\w*|other uses|different meanings|films?|movies?)\b',query,re.I) and re.search(r'may also refer|topics referred to by the same term',text,re.I):return False
    overlap=query_overlap(query,text)
    if allow_ranked and passage.metadata.get('upstream_passage_id') and passage.metadata.get('task4_retrieval_mode')=='lexical':
        # Task 4 already ranked this query-local collection. TF-IDF magnitude or
        # cross-language lexical overlap cannot decide semantic answerability.
        return True
    topic=passage.metadata.get('task5_topic',{})
    if topic.get('document_anchor') and topic.get('strong_match') and len(match_tokens(text)&set(topic.get('context_terms',[])))>=2:return True
    # Strong semantic handoffs can legitimately use synonyms or another language.
    # Weak lexical handoffs must match the topic at the sentence level.
    return not query_terms(query) or overlap>0 or passage.query_rel>=.70

def introductory_statement(query,text):
    return bool(definition_match(query,text) and query_overlap(query,' '.join(text.split()[:4])) and
                re.search(r'\b(?:is|are|was|were)\s+(?:an?|one|the|among)\b',text,re.I))
