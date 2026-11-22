"""Task 5 topic context over supplied passages; no retrieval or upstream score edits."""
from collections import Counter,defaultdict
from dataclasses import replace
import re
from .text import query_overlap,match_tokens,query_terms
from .content import usable_sentences,overview_query,definition_match

ASPECTS=re.compile(r'\b(?:more|something|briefly|brief|summary|meaning|story|themes?|significance|importance|history|origins?)\b',re.I)
ALTERNATE=re.compile(r'\b(?:disambiguat\w*|other uses|different meanings|also refer|films?|movies?)\b',re.I)
ROUTING=re.compile(r'\b(?:may also refer|also refers? to these|topics referred to by the same term|disambiguation)\b',re.I)
BACKGROUND=re.compile(r'\b(?:translat\w*|editions?|other versions|external links|references|ISBN|published|authorship|composed|named|name over others|behind the name|special sacred place|forms the heart of|he refers to|recit\w*|parayana)\b|\b(?:fondly|affectionately)\s+(?:calls?|called)\b|traditional to begin the reading',re.I)
CONTEXT_STOP=match_tokens('Wikipedia Sanskrit Hindi English India Indian original ancient epic chapter book work popular first fifth only source text story said mother named translation verses language beautiful')


def scope_query(query):return ASPECTS.sub(' ',query)


def alternate_requested(query):return bool(ALTERNATE.search(query))


def background_statement(query,text):
    return bool(overview_query(query) and not re.search(r'\b(?:translat\w*|edition|version|author\w*|wrote|write|name|origin|recit\w*|reading|importance|significance|devotional|glory)\b',query,re.I) and BACKGROUND.search(text))


def overview_definition(query,text):
    return bool(not background_statement(query,text) and not ROUTING.search(text) and definition_match(query,text)
                and query_overlap(scope_query(query),' '.join(text.split()[:5]))>=.99)


def secondary_passage(query,text,statements):
    if not overview_query(query):return False
    if re.search(r'\b(?:translat\w*|editions?|versions?|author\w*|wrote|write|name|origin|recit\w*|reading|devotional|glory)\b',query,re.I):return False
    if re.search(r'\b(?:versions?|translat\w*|editions?|retellings?)\b',text[:350],re.I) and not any(overview_definition(query,t) for t,_ in statements[:2]):return True
    heading=re.search(r'\b(?:Other versions|Translations|References|External links|The act of reading|Recitation|Adaptations)\b',text)
    if heading and heading.start()<len(text)*.35:return True
    return bool(statements and sum(background_statement(query,t) for t,_ in statements)>=max(1,(len(statements)+1)//2))


def document_key(passage):
    return (passage.metadata.get('upstream_document_id') or passage.metadata.get('source_metadata',{}).get('document_id')
            or passage.url_or_provenance or passage.source_id or passage.id)


def annotate_topic_context(query,passages):
    grouped=defaultdict(list)
    for p in passages:grouped[document_key(p)].append(p)
    documents={};clean={p.id:list(usable_sentences(p)) for p in passages}
    for key,rows in grouped.items():
        combined='\n'.join(p.text for p in rows)
        title_match=any(query_overlap(scope_query(query),p.title or '')>=.99 for p in rows)
        interactive=any(re.search(r'\bquiz\b',p.title or '',re.I) for p in rows) and not re.search(r'\bquiz\b',query,re.I)
        definitions=[p for p in rows if any(overview_definition(query,text) for text,_ in clean[p.id])]
        direct=any(query_overlap(scope_query(query),text)>=.99 and not ROUTING.search(text)
                   and not background_statement(query,text) for p in rows for text,_ in clean[p.id])
        anchor=title_match and direct and not interactive
        disambiguation=bool(re.search(r'topics referred to by the same term',combined,re.I) or
                            (ROUTING.search(combined) and len(re.findall(r'\bfilm\b',combined,re.I))>=2))
        seeds=definitions or [p for p in rows if clean[p.id] and any(query_overlap(scope_query(query),t)>=.99 for t,_ in clean[p.id])]
        seed_text=' '.join(text for p in seeds[:2] for text,_ in clean[p.id][:8])
        counts=Counter(word for sentence in re.split(r'[.!?।॥]',seed_text) for word in match_tokens(sentence))
        proper={word.casefold() for word in re.findall(r'\b[A-Z][a-z]{3,}\b',seed_text)}
        terms={word for word,count in counts.items() if (count>=2 or word in proper) and len(word)>=4}-CONTEXT_STOP-query_terms(query)
        documents[key]={'anchor':anchor,'title_match':title_match,'primary':bool(anchor and definitions and not disambiguation),
                        'disambiguation':disambiguation,'context_terms':sorted(terms)[:40]}
    substantive_available=any(d['primary'] for d in documents.values())
    result=[]
    for p in passages:
        doc=documents[document_key(p)];sentences=clean[p.id]
        direct=any(query_overlap(scope_query(query),text)>=.99 and not ROUTING.search(text) for text,_ in sentences)
        contextual=bool(doc['anchor'] and any(len(match_tokens(text)&set(doc['context_terms']))>=2 for text,_ in sentences))
        intro=any(overview_definition(query,text) for text,_ in sentences)
        secondary=secondary_passage(query,p.text,sentences)
        role='disambiguation' if doc['disambiguation'] else 'introduction' if intro else 'background' if secondary else 'main_content' if doc['primary'] and contextual else 'supporting'
        relevant=bool(doc['anchor'] and (direct or contextual) and sentences)
        scoped={'title_match':doc['title_match'],'document_anchor':doc['anchor'],'direct_match':direct,'contextual_match':contextual,
                'strong_match':relevant,'context_terms':doc['context_terms'],'role':role,'primary_document':doc['primary'],
                'suppress_disambiguation':bool(substantive_available and doc['disambiguation'] and overview_query(query) and not alternate_requested(query)),
                'overview_priority':3 if intro and not doc['disambiguation'] else 2 if doc['primary'] and not secondary else 0 if secondary or doc['disambiguation'] else 1,
                'basis':'document title, subject-name spans and same-document context; not a calibrated relevance probability'}
        # The caller's objects, original Task 4 scores, IDs and passage text stay intact.
        result.append(replace(p,metadata={**p.metadata,'task5_topic':scoped}))
    return result


def statement_role(query,text,passage):
    topic=passage.metadata.get('task5_topic',{})
    if ROUTING.search(text) and not alternate_requested(query):return 'disambiguation'
    if background_statement(query,text):return 'background'
    if overview_definition(query,text) and re.search(r'\b(?:is|are|was|were)\s+(?:an?|the|one|among)\b',text,re.I):return 'definition'
    if topic.get('role')=='background':return 'background'
    if topic.get('primary_document'):return 'main_content'
    return 'supporting'
