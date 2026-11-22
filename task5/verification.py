"""Separate retrieval diagnostics and bounded cross-sample consistency signals."""
import statistics
from .text import normalize


def retrieval_diagnostics(candidates,selected):
    scores=[p.query_rel for p in candidates]
    if not scores:return {'status':'no_evidence','received':0,'selected':len(selected),'low_relevance':True,'flat_scores':False}
    ordered=sorted(scores,reverse=True);leading=ordered[:min(10,len(ordered))]
    spread=max(leading)-min(leading)
    return {'status':'measured','received':len(candidates),'selected':len(selected),
            'maximum_query_relevance':round(max(scores),4),'mean_query_relevance':round(statistics.mean(scores),4),
            'leading_score_range':round(spread,4),'top_two_margin':round(ordered[0]-ordered[1],4) if len(scores)>1 else None,
            'low_relevance':max(scores)<.35,'flat_scores':len(leading)>=3 and spread<.02,
            'document_topic_match':bool(selected) and all(p.metadata.get('task5_topic',{}).get('strong_match') for p in selected),
            'topic_matched_passages':sum(bool(p.metadata.get('task5_topic',{}).get('strong_match')) for p in selected),
            'interpretation':'Task 4 ranking features are uncalibrated scores. Low lexical scores or flatness alone do not prove irrelevant or insufficient evidence.'}


def consistency_signal(claims,samples,config):
    """A second summary is a stability signal, not independent factual evidence."""
    from .nli import readiness,load_model
    available=readiness(config.nli_directory)['ready']
    try:model=load_model(config.nli_directory) if available else None
    except Exception:model=None
    rows=[]
    for index,claim in enumerate(claims):
        survived=0
        for sample in samples:
            matches=False
            for alternative in sample:
                if normalize(alternative.text).casefold()==normalize(claim.text).casefold():matches=True;break
                if model:
                    score=model.score(alternative.text,claim.text)
                    if not score['truncated'] and score['entailment']>=config.nli_entailment_threshold:
                        matches=True;break
            survived+=int(matches)
        rows.append({'claim_index':index,'survived_samples':survived,'sample_count':len(samples),
                     'survival_fraction':survived/len(samples) if samples else None,'unstable':bool(samples) and survived<len(samples)})
    return {'status':'completed' if samples else 'not_run','additional_drafts':len(samples),
            'matching_method':'multilingual_nli_entailment' if model else 'exact_text_only',
            'claims':rows,'interpretation':'Small-sample recurrence is not a calibrated confidence estimate or proof of truth.'}
