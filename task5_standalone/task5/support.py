"""Model-judged evidence support with deterministic citation/conflict/OCR guards."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def support_policy():
    fallback={'schema_version':'task5.support_policy.v1','high_requires_nli_entailment':True,'high_allow_unknown_source_with_nli':False,
              'high_blocked_judge_models':['qwen3:4b-instruct-2507-q4_K_M'],
              'calibration':{'status':'unavailable','human_review_attested':False}}
    try:
        path=Path(__file__).with_name('support_policy.json')
        if path.stat().st_size>50000:return fallback
        value=json.loads(path.read_text())
        if value.get('schema_version')!=fallback['schema_version'] or type(value.get('high_requires_nli_entailment')) is not bool or type(value.get('high_allow_unknown_source_with_nli',False)) is not bool:return fallback
        calibration=value.get('calibration')
        blocked=value.get('high_blocked_judge_models',[])
        if not isinstance(blocked,list) or len(blocked)>20 or any(not isinstance(v,str) or len(v)>200 for v in blocked):return fallback
        if not isinstance(calibration,dict) or calibration.get('status') not in {'prototype','model_judged_pilot'} or calibration.get('human_review_attested') is not False:return fallback
        return value
    except (OSError,ValueError,TypeError):return fallback

SUPPORT_SCHEMA={'type':'object','properties':{
    'label':{'type':'string','enum':['High','Medium','Low']},
    'grounding':{'type':'string','enum':['strong','partial','weak']},
    'question_coverage':{'type':'string','enum':['complete','partial','not_answered']},
    'source_quality':{'type':'string','enum':['adequate','weak','unknown']},
    'evidence_agreement':{'type':'string','enum':['consistent','conflicting','not_assessable']},
    'explanation':{'type':'string','maxLength':500}},
    'required':['label','grounding','question_coverage','source_quality','evidence_agreement','explanation'],
    'additionalProperties':False}


def validate_assessment(value):
    if not isinstance(value,dict):return False
    for key,schema in SUPPORT_SCHEMA['properties'].items():
        if key=='explanation':
            if not isinstance(value.get(key),str) or not 1<=len(value[key])<=500:return False
        elif value.get(key) not in schema['enum']:return False
    return True


def assess_support(claims,used,conflicts,review,quote_coverage):
    """High describes support in references, not a probability of factual truth.

    Prototype calibration warnings and the absence of repeated wording do not
    force every well-supported paraphrase to Low. A judge cannot override guards.
    """
    assessment=review.get('support_assessment')
    if review.get('status')!='completed' or not validate_assessment(assessment):return None
    reasons=[];criteria=review.get('criteria',{});nli=review.get('nli',{});policy=support_policy()
    accepted=set(nli.get('accepted_claim_indices',range(len(claims))))
    contradiction=any(r.get('verdict')=='contradicted' for r in nli.get('claims',[]) if r.get('claim_index') in accepted)
    ocr=[p.ocr_conf for p in used if p.is_ocr()]
    poor_ocr=any(value is not None and value<.65 for value in ocr)
    incomplete_ocr=any(value is None or value<.85 for value in ocr)
    relevant=criteria.get('answer_relevance')=='pass'
    complete=(assessment['question_coverage']=='complete' and criteria.get('evidence_coverage')=='pass'
              and review.get('point_count',{}).get('complete',True))
    reduced=review.get('reviewed_claims',len(claims))>len(claims)
    stable=not any(row.get('unstable') for row in review.get('consistency',{}).get('claims',[]) if row.get('claim_index') in accepted)
    entailed={r.get('claim_index') for r in nli.get('claims',[]) if r.get('verdict')=='entailed'}
    nli_high=not policy['high_requires_nli_entailment'] or nli.get('status')=='completed' and accepted<=entailed
    verified_unknown=(assessment['source_quality']=='unknown' and policy.get('high_allow_unknown_source_with_nli',False)
                      and nli.get('status')=='completed' and accepted<=entailed
                      and bool(used) and all(p.url_or_provenance or p.publisher or p.source_id for p in used))
    adequate_source=assessment['source_quality']=='adequate' or verified_unknown
    if not claims or quote_coverage<1:reasons.append('accepted claims do not all have validated supporting quotes')
    if contradiction or conflicts or assessment['evidence_agreement']=='conflicting':reasons.append('conflicting evidence remains unresolved')
    if criteria.get('answer_relevance') in {'fail','unassessable'} or assessment['question_coverage']=='not_answered':reasons.append('the answer does not fulfill the question')
    if assessment['grounding']=='weak':reasons.append('the judge found weak claim grounding')
    if poor_ocr:reasons.append('cited OCR confidence is low')
    if reasons:label='LOW'
    elif (assessment['label']=='High' and assessment['grounding']=='strong' and relevant and complete and not reduced
          and adequate_source and not incomplete_ocr and stable and nli_high):label='HIGH'
    else:
        label='LOW' if assessment['label']=='Low' else 'MEDIUM'
        if not complete or reduced:reasons.append('question coverage is partial')
        if assessment['source_quality']!='adequate':reasons.append('source quality needs review')
        if incomplete_ocr:reasons.append('OCR quality is moderate or unknown')
        if not stable:reasons.append('claims varied in the consistency sample')
        if not nli_high:reasons.append('independent entailment does not meet the fitted High-support rule')
    if label=='HIGH' and review.get('judge_model') in policy.get('high_blocked_judge_models',[]):
        label='MEDIUM';reasons.append('this local judge has not validated High support; use the configured stronger review model')
    return {'label':label,'judge_proposed_label':assessment['label'],'assessment':assessment,'guard_reasons':reasons,
            'basis':'LLM-judged support in the supplied references plus citation, contradiction, coverage and OCR guards',
            'calibration_status':policy['calibration']['status'],'calibration':policy['calibration'],
            'truth_probability':None}
