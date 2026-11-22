"""Bounded MMR selection and conservative, cached lexical conflict checks."""
from collections import Counter
from itertools import combinations
import re
from .text import NEGATIONS, duplicate_similarity, indian_query, normalize, sentences, similarity, tokens
from .content import usable_sentences,definition_match
from .topic import annotate_topic_context
from .intent import focus_score

AUTHORITY={'High':1.0,'Medium':.65,'Low':.20,'Unknown':.35}


def authority(p):return AUTHORITY[p.parsed_source_signal().value]


def select_evidence(query,passages,config):
    passages=annotate_topic_context(query,passages)
    eligible,decisions=[],{}
    for p in passages:
        reason=''
        if not normalize(p.text):reason='empty text'
        elif p.is_ocr() and p.ocr_conf is not None and p.ocr_conf<config.min_ocr_conf:reason='low OCR confidence'
        elif len(p.text)>config.max_passage_chars:reason='passage exceeds context limit; split upstream'
        elif not any(usable_sentences(p)):reason='no usable answer statements; navigation, questions or fragments only'
        elif p.metadata['task5_topic']['suppress_disambiguation']:reason='disambiguation page; a substantive topic article is available'
        elif p.query_rel<config.min_query_rel and not (p.metadata.get('upstream_passage_id') and
             (config.provider!='extractive' and p.metadata.get('task4_retrieval_mode')=='lexical' or p.metadata.get('task4_retrieval_mode')!='semantic' and p.metadata['task5_topic']['strong_match'])):reason='below Task 5 query relevance threshold without a verified document-topic match'
        if reason:
            decisions[p.id]={'id':p.id,'selected':False,'reason':reason};continue
        context=config.context_weight if indian_query(query) else 0.0
        relevance=(1-context)*p.query_rel+context*p.india_rel
        ocr=p.ocr_conf if p.ocr_conf is not None else (.45 if p.is_ocr() else 1.0)
        priority=relevance*(.4+.6*authority(p))*(.5+.5*ocr)
        # Broad overview queries need introductory statements. Otherwise repeated
        # topic names in references can consume the source cap before the lead.
        priority+=.35*max((definition_match(query,text) for text,_ in usable_sentences(p)),default=0)
        priority+=.45*max((focus_score(query,text) for text,_ in usable_sentences(p)),default=0)
        if p.metadata['task5_topic']['strong_match']:priority+=.12*p.metadata['task5_topic']['overview_priority']
        eligible.append((p,priority))
    eligible.sort(key=lambda item:(-item[1],item[0].id))
    unique=[]
    for p,score in eligible:
        duplicate=next((q.id for q,_ in unique if duplicate_similarity(p.text,q.text)>=config.duplicate_threshold),None)
        if duplicate:decisions[p.id]={'id':p.id,'selected':False,'reason':'copy/near-duplicate of '+duplicate}
        else:unique.append((p,score))
    remaining=list(unique);selected=[];counts=Counter();chars=0
    while remaining and len(selected)<config.max_passages:
        def mmr(item):
            p,priority=item
            return config.mmr_lambda*priority-(1-config.mmr_lambda)*max((similarity(p.text,q.text) for q in selected),default=0)
        p,score=max(remaining,key=lambda item:(mmr(item),item[1],item[0].id));remaining.remove((p,score))
        identity=p.source_identity() or 'unknown:'+p.id
        reason=''
        if counts[identity]>=config.max_per_source:reason='source group cap'
        elif chars+len(p.text)>config.context_chars:reason='total context budget'
        elif authority(p)<=.2 and any(authority(q)>=.65 for q in selected):reason='weak source cannot dominate stronger evidence'
        if reason:decisions[p.id]={'id':p.id,'selected':False,'reason':reason};continue
        selected.append(p);counts[identity]+=1;chars+=len(p.text)
        rescued=p.query_rel<config.min_query_rel
        decisions[p.id]={'id':p.id,'selected':True,'reason':('Task 4 lexical handoff retained; raw scores are not calibrated relevance cutoffs' if p.metadata.get('task4_retrieval_mode')=='lexical' else 'document-topic match retained low lexical-score evidence') if rescued else 'query relevance, document-topic match, source/OCR quality and MMR diversity',
                         'priority':round(score,4),'original_query_relevance':p.query_rel,'topic':p.metadata['task5_topic']}
    for p,_ in remaining:decisions[p.id]={'id':p.id,'selected':False,'reason':'passage count limit'}
    return selected,[p for p,_ in unique],[decisions[p.id] for p in passages]


def detect_conflicts(passages):
    records={}
    for p in passages[:50]:
        records[p.id]=[]
        for sentence in sentences(p.text)[:8]:
            words=tokens(sentence)
            clean={w[:-2] if w.isascii() and len(w)>5 and w.endswith('ed') else w
                   for w in words-NEGATIONS-{'did'} if not w.isdigit()}
            records[p.id].append((sentence,clean,bool(words&NEGATIONS),set(re.findall(r'\b\d+(?:\.\d+)?\b',sentence))))
    conflicts=[]
    for a,b in combinations(passages[:50],2):
        found=False
        for sa,ta,nega,na in records[a.id]:
            for sb,tb,negb,nb in records[b.id]:
                shared,union=ta&tb,ta|tb
                if len(shared)<3 or not union or len(shared)/len(union)<.65:continue
                # Overlapping chunks can quote the same statement with an extra
                # date/identifier in its lead-in. A contained affirmative quote
                # is not a numeric contradiction with the longer version.
                if nega==negb and (normalize(sa) in normalize(sb) or normalize(sb) in normalize(sa)):continue
                if nega!=negb or (na and nb and na!=nb):
                    conflicts.append({'passage_ids':[a.id,b.id],'quotes':[sa,sb],
                        'description':'Potential incompatible numbers or negation in closely matching statements', 'method':'lexical_heuristic'})
                    found=True;break
            if found:break
        if len(conflicts)>=8:break
    return conflicts


def scope_answer_conflicts(conflicts,claims,query):
    """Keep unrelated numeric disputes as corpus diagnostics, not answer limits."""
    from .intent import question_kind
    digits=lambda text:set(re.findall(r'\d+',text.translate(str.maketrans('०१२३४५६७८९','0123456789'))))
    answer_numbers=digits(' '.join(c.text for c in claims))
    relevant=[];background=[]
    for conflict in conflicts:
        quotes=conflict.get('quotes',[])
        if len(quotes)==2:
            left,right=(digits(quote) for quote in quotes)
            differences=left^right
            if differences and not answer_numbers&differences and question_kind(query)!='date':
                background.append(conflict);continue
        relevant.append(conflict)
    return relevant,background
