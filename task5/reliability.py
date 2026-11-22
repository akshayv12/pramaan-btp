"""Interpretable support labels. Context priority is never treated as source trust."""
from .evidence import authority
from .schemas import ReliabilityRating
from .text import normalize
from .content import concise_statement
from .support import assess_support


def assess(claims,used,pool,conflicts,review):
    if not claims:
        return ReliabilityRating.LOW,'No usable supporting evidence was available; the system abstained.',{
            'source_quality':0,'query_relevance':0,'context_relevance':0,'ocr_quality':None,'evidence_coverage':0,
            'cross_source_agreement':0,'documented_source_groups':0,'unresolved_contradictions':bool(conflicts),'limitations':['No accepted claims']}
    by_id={p.id:p for p in used}
    valid=sum(bool(c.evidence) and all(e.passage_id in by_id and normalize(e.quote) in normalize(by_id[e.passage_id].text)
        and (normalize(c.text)==concise_statement(e.quote) if c.support_method=='pronunciation_metadata_omission'
             else c.support_method!='exact_extraction' or normalize(c.text) in normalize(e.quote)) for e in c.evidence) for c in claims)
    coverage=valid/len(claims)
    groups={p.source_identity() for p in used}-{None}
    source_quality=sum(authority(p) for p in used)/len(used) if used else 0
    relevance=sum(p.query_rel for p in used)/len(used) if used else 0
    context=sum(p.india_rel for p in used)/len(used) if used else 0
    topic_matched=bool(used) and all(p.metadata.get('task5_topic',{}).get('strong_match') for p in used)
    ocr=[p.ocr_conf for p in used if p.is_ocr()]
    corroborated=0
    for c in claims:
        source_groups={p.source_identity() for p in pool if normalize(c.text).casefold() in normalize(p.text).casefold()}-{None}
        if len(source_groups)>=2:corroborated+=1
    agreement=corroborated/len(claims)
    limitations=[];rating=ReliabilityRating.MEDIUM
    if coverage<1:limitations.append('incomplete claim attribution');rating=ReliabilityRating.LOW
    if source_quality<.45 or all(authority(p)<=.35 for p in used):limitations.append('source authority is weak or unestablished');rating=ReliabilityRating.LOW
    if relevance<.35:
        if topic_matched:limitations.append('upstream relevance scores are low, although document-topic context matches the query')
        elif review.get('status')=='completed' and review.get('criteria',{}).get('answer_relevance')=='pass':
            limitations.append('upstream relevance scores are low; the model judged the cited answer relevant');rating=ReliabilityRating.LOW
        else:limitations.append('the cited evidence only weakly matches the query');rating=ReliabilityRating.LOW
    if ocr and any(v is None or v<.70 for v in ocr):limitations.append('low or unknown OCR confidence');rating=ReliabilityRating.LOW
    if len(groups)<2:limitations.append('fewer than two documented source groups')
    if any(p.source_identity() is None for p in used):limitations.append('source provenance is incomplete')
    independent=bool(used) and all(p.metadata.get('independence_verified') is True for p in used)
    if not independent:limitations.append('source independence is not verified')
    if agreement<1:limitations.append('not every claim has exact cross-source corroboration')
    if conflicts:limitations.append('potential contradictions remain unresolved')
    if any(c.support_method not in {'exact_extraction','pronunciation_metadata_omission'} for c in claims):limitations.append('semantic support relies on an uncalibrated model review')
    nli=review.get('nli',{});consistency=review.get('consistency',{});retrieval=review.get('retrieval',{})
    accepted=set(nli.get('accepted_claim_indices',[]))
    if review.get('profile') in {'standard','thorough'}:
        if nli.get('status')!='completed':limitations.append('independent NLI verification was unavailable')
        elif any(row['verdict']!='entailed' for row in nli.get('claims',[]) if row['claim_index'] in accepted):
            limitations.append('some claims were borderline for NLI and accepted by the model judge')
        if nli.get('thresholds',{}).get('calibration_status')!='reviewed_holdout_evaluated':limitations.append('verification thresholds lack language-specific human calibration')
        if any(row.get('unstable') for row in consistency.get('claims',[]) if row['claim_index'] in accepted):
            limitations.append('some claims did not recur in the sampled alternative answer');rating=ReliabilityRating.LOW
        if consistency.get('status')=='not_run':limitations.append('sampling consistency was not measured in standard mode')
        if review.get('criteria',{}).get('evidence_coverage') in {'partial','fail','unassessable'}:limitations.append('the judge could not establish complete question coverage')
    if retrieval.get('low_relevance') and not retrieval.get('document_topic_match'):
        limitations.append('retrieved candidates have low uncalibrated relevance scores');rating=ReliabilityRating.LOW
    if retrieval.get('flat_scores'):limitations.append('leading retrieval scores are closely clustered; their order is weakly distinguished')
    if any(p.metadata.get('synthetic') for p in used):limitations.append('evidence is an illustrative test fixture')
    if ocr and any(v is not None and v<.85 for v in ocr):limitations.append('OCR confidence is moderate')
    if not limitations and source_quality>=.85 and relevance>=.75 and coverage==1 and len(groups)>=2:
        rating=ReliabilityRating.HIGH
    judged=assess_support(claims,used,conflicts,review,coverage)
    if judged:rating=ReliabilityRating(judged['label'])
    reasons=[f'{valid}/{len(claims)} claims have checked source quotes from {len(groups)} documented source group(s).']
    if ocr:reasons.append('Original OCR confidence is retained.')
    else:reasons.append('Cited evidence is direct digital text.')
    if limitations:reasons.append('Support is limited because '+ '; '.join(limitations)+'.')
    if not conflicts:
        reasons.append('No conflict was found by '+('the combined model and limited lexical checks' if review['status']=='completed' else 'limited lexical checks')+'; this does not prove agreement.')
    if judged:
        reasons=[f'{rating.value.title()} support in the supplied sources: '+judged['assessment']['explanation']]
        if judged['guard_reasons']:reasons.append('Limits: '+'; '.join(judged['guard_reasons'])+'.')
        reasons.append(f'{valid}/{len(claims)} claims have validated source quotes. This is a model-judged support label, not a truth probability.')
    factors={'source_quality':round(source_quality,3),'query_relevance':round(relevance,3),
        'context_relevance':round(context,3),'context_increases_trust':False,'document_topic_match':topic_matched,'ocr_quality':min(ocr) if ocr and all(v is not None for v in ocr) else None,
        'evidence_coverage':coverage,'coverage_definition':'accepted claims with checked quotes; not completeness against an unknown gold answer',
        'cross_source_agreement':agreement,'agreement_method':'exact claim occurrence across documented groups; not semantic proof',
        'documented_source_groups':len(groups),'independence_verified':independent,'unresolved_contradictions':bool(conflicts),
        'semantic_review':review,'retrieval_quality':retrieval,'faithfulness_verifier':nli,
        'sampling_consistency':consistency,'judge_criteria':review.get('criteria',{}),
        'limitations':limitations,'llm_support_assessment':judged,
        'source_quality_basis':'declared source category; separate from the LLM source-quality judgment',
        'assessment':'LLM-judged reference support with deterministic guards; not probability of truth' if judged else 'heuristic evidence support, not probability of truth'}
    return rating,' '.join(reasons),factors
