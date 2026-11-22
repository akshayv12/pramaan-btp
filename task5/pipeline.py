"""Task 4 evidence -> diverse query-local collection -> cited answer + support label."""
from pathlib import Path
from dataclasses import replace
import os
from time import perf_counter
from .collection import build_collection
from .config import Task5Config,DEFAULT_GEMINI_MODEL
from .evidence import detect_conflicts,select_evidence,scope_answer_conflicts
from .generation import synthesize
from .providers import ProviderError,StructuredProvider
from .reliability import assess
from .schemas import EvidencePassage,SummaryResult
from .language import resolve_answer_language
from .intent import requested_count
from .verification import retrieval_diagnostics


def coerce_evidence(values):
    if not isinstance(values,(list,tuple)) or len(values)>500:raise ValueError('Evidence must be a list of at most 500 passages')
    passages=[p if isinstance(p,EvidencePassage) else EvidencePassage.from_dict(p) for p in values]
    if len({p.id for p in passages})!=len(passages):raise ValueError('Passage IDs must be unique within a query')
    return passages


class Task5Pipeline:
    def __init__(self,config=None,*,provider=None):
        self.config=config or Task5Config.from_env()
        self.provider=provider

    def run(self,query,ranked_evidence,target_language=None):
        if not isinstance(query,str) or not 1<=len(query.strip())<=2000:raise ValueError('A nonempty query of at most 2000 characters is required')
        if target_language not in {None,'hi','en'}:raise ValueError('target_language must be hi or en')
        start=perf_counter();language_resolution=resolve_answer_language(query,target_language)
        language=language_resolution['resolved_language']
        count=requested_count(query)
        if count:
            # Honour a bounded list request without enlarging ordinary answers.
            scoped=replace(self.config,max_claims=min(count,8))
            if scoped!=self.config:
                return Task5Pipeline(scoped,provider=self.provider).run(query,ranked_evidence,target_language)
        candidates=coerce_evidence(ranked_evidence)
        selected,pool,decisions=select_evidence(query,candidates,self.config)
        provider=self.provider;provider_error=None
        if selected and provider is None and self.config.provider!='extractive':
            try:provider=StructuredProvider(self.config)
            except ProviderError as exc:
                if not self.config.allow_fallback:raise
                provider_error=str(exc)
        planning={'status':'not_run'};planning_calls=[]
        if selected and provider is not None and self.config.query_focused_selection:
            from .planning import select_for_question
            try:
                planner=provider
                if self.config.judge_provider=='gemini' and provider.name!='gemini':
                    planner=StructuredProvider(replace(self.config,provider='gemini',model=os.getenv('GEMINI_MODEL',DEFAULT_GEMINI_MODEL)))
                selected,planning=select_for_question(query,pool,selected,language,self.config,planner)
                if planner is not provider:planning_calls=list(planner.calls)
                chosen={p.id for p in selected}
                for decision in decisions:
                    if decision['id'] in {p.id for p in pool}:
                        decision.update(selected=decision['id'] in chosen,reason='query-focused selection from existing Task 4 candidates')
            except ProviderError as exc:
                if not self.config.allow_fallback:raise
                planning={'status':'failed','reason':str(exc)}
        collection=build_collection(query,selected)
        claims,mode,warnings,model_conflicts,review=synthesize(query,selected,pool,language,self.config,provider)
        review['evidence_planning']=planning
        if planning.get('answerability')=='insufficient':warnings.append('The evidence selector found the requested information absent: '+planning['reason'])
        review['retrieval']=retrieval_diagnostics(candidates,selected)
        if provider_error:
            mode='extractive_fallback';warnings.append(provider_error+'; returned an explicit extractive fallback.')
        cited=list(dict.fromkeys(e.passage_id for c in claims for e in c.evidence))
        used=[p for p in selected if p.id in cited]
        conflicts=detect_conflicts(pool)
        for conflict in model_conflicts:
            if not any(c['passage_ids']==conflict['passage_ids'] and c['quotes']==conflict['quotes'] for c in conflicts):conflicts.append(conflict)
        conflicts,background_conflicts=scope_answer_conflicts(conflicts,claims,query)
        if count:
            review['point_count']={'requested':count,'accepted':len(claims),'complete':len(claims)>=count}
        rating,why,factors=assess(claims,used,pool,conflicts,review)
        if mode=='extractive_fallback':why+=' Model generation or verification failed; this answer uses source extraction.'
        status='answered' if claims else 'insufficient_evidence'
        topic_matched=bool(used) and all(p.metadata.get('task5_topic',{}).get('strong_match') for p in used)
        criteria=review.get('criteria',{})
        judged_coverage=criteria.get('evidence_coverage') if review.get('status')=='completed' else None
        if judged_coverage in {'pass','partial','fail','unassessable'}:
            # The rubric concerns the actual question and cited answer. Raw,
            # uncalibrated ranking-feature magnitudes are a separate diagnostic.
            limited=bool(claims) and (judged_coverage!='pass' or
                criteria.get('answer_relevance') in {'partial','fail','unassessable'} or
                review.get('reviewed_claims',len(claims))>len(claims))
        else:
            limited=bool(claims) and max((p.query_rel for p in used),default=0)<.35 and not topic_matched
        if claims and mode.startswith('extractive') and any(p.language!=language for p in used):limited=True
        if count and claims and len(claims)<count:
            limited=True
            warnings.append(f'The question requests {count} distinct points; only {len(claims)} supported points were accepted.')
            if rating.value=='HIGH':
                from .schemas import ReliabilityRating
                rating=ReliabilityRating.MEDIUM
                why+=' The accepted answer does not provide all requested points.'
        if limited:status='limited'
        factors['answer_coverage']={'status':status,'topic_matched':topic_matched,
                                   'judge_coverage':review.get('criteria',{}).get('evidence_coverage'),
                                   'interpretation':'Answer coverage is separate from source reliability and raw ranking-feature magnitude.'}
        statements=[c.text+' ['+', '.join(dict.fromkeys(e.passage_id for e in c.evidence))+']' for c in claims]
        answer='\n'.join(f'{index}. {text}' for index,text in enumerate(statements,1)) if count else ' '.join(statements)
        if not answer:answer='इस प्रश्न का उत्तर देने के लिए पर्याप्त उपयोगी साक्ष्य उपलब्ध नहीं है।' if language=='hi' else 'There is insufficient usable evidence to answer this question.'
        elif limited:
            answer=('सीमित साक्ष्य: ये अंश प्रश्न का पूरा उत्तर स्थापित नहीं करते। ' if language=='hi' else 'Limited evidence: these excerpts do not establish a complete answer. ')+ ('\n' if count else '')+answer
        if conflicts and claims:
            answer+=(' स्रोतों में संभावित विरोध है; मूल अंशों की जाँच करें।' if language=='hi' else ' Potential source disagreement remains; inspect the source excerpts.')
        for d in decisions:
            if 'OCR' in d['reason'] and not d['selected']:warnings.append(d['id']+': '+d['reason'])
        if len(pool)>50:warnings.append('Lexical conflict checks cover the first 50 eligible passages and eight sentences per passage.')
        supporting=[]
        for p in used:
            supporting.append({'id':p.id,'original_passage_id':p.metadata.get('upstream_passage_id',p.id),
                'title':p.title,'url':p.url_or_provenance,'publisher':p.publisher,'page':p.page,
                'extraction_method':p.extraction_method,'ocr_confidence':p.ocr_conf,
                'quotes':list(dict.fromkeys(e.quote for c in claims for e in c.evidence if e.passage_id==p.id))})
        calls=planning_calls+(list(getattr(provider,'calls',[])) if provider is not None else [])
        calls+=review.get('judge_calls',[])
        diagnostics={'package_path':str(Path(__file__).resolve().parent),'duration_ms':round((perf_counter()-start)*1000,3),
            'config':self.config.to_dict(),'selection':decisions,'collection':{k:v for k,v in collection.items() if k!='documents'},
            'document_count':len(collection['documents']),'model_calls':calls if isinstance(calls,list) else [],
            'review':review,'background_conflicts':background_conflicts,'language_resolution':language_resolution,
            'limitations':['Support labels concern supplied evidence, not truth probabilities.','Context relevance guides selection independently of source trust.']}
        return SummaryResult(query,answer,language,cited,rating,why,factors,used,claims,supporting,selected,conflicts,warnings,mode,status,diagnostics)


def summarize_with_reliability(query,ranked_evidence,target_language=None,*,config=None):
    return Task5Pipeline(config=config).run(query,ranked_evidence,target_language)
