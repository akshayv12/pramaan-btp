"""Small Task 4 replay evaluation; human quality/accuracy is never inferred from quotes."""
import argparse
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
from .config import Task5Config
from .handoff import adapt_task4_envelope,summarize_task4_payload,write_json_atomic
from .text import duplicate_similarity,normalize


def evaluate(payload,config=None):
    result=summarize_task4_payload(payload,config=config or Task5Config(provider='extractive',min_query_rel=.10))
    envelopes=payload.get('task4_output',{}).get('queries',[payload]);rows=[]
    for envelope,answer in zip(envelopes,result['task5_output']['queries']):
        passages,_=adapt_task4_envelope(envelope);by_id={p.id:p for p in passages}
        r=answer['result'];claims=r['claims'];refs=[e for c in claims for e in c['evidence']]
        valid=sum(e['passage_id'] in by_id for e in refs)
        quote_valid=sum(e['passage_id'] in by_id and normalize(e['quote']) in normalize(by_id[e['passage_id']].text) for e in refs)
        pairs=list(combinations(claims,2))
        rows.append({'query':answer['query'],'is_simulated':answer['is_simulated'],'answer_status':answer['answer_status'],
            'reliability':r['reliability'],'generation_mode':r['generation_mode'],'claims':len(claims),'citations':len(refs),
            'citation_id_validity':valid/len(refs) if refs else None,'exact_quote_validity':quote_valid/len(refs) if refs else None,
            'redundant_claim_pair_fraction':sum(duplicate_similarity(a['text'],b['text'])>=.76 for a,b in pairs)/len(pairs) if pairs else None,
            'documented_source_groups':r['factors']['documented_source_groups'],'duration_ms':r['diagnostics']['duration_ms'],
            'answer_characters':len(r['answer']),'human_coverage':None,'human_relevance':None,'human_readability':None,
            'semantic_factual_support':None,'human_reliability_explanation':None})
    return {'schema_version':'task5.replay_evaluation.v1','cases':rows,
        'aggregate':{'queries':len(rows),'mean_duration_ms':mean(r['duration_ms'] for r in rows) if rows else None,
            'citation_metric_cases':sum(r['citation_id_validity'] is not None for r in rows),
            'mean_citation_id_validity':mean(r['citation_id_validity'] for r in rows if r['citation_id_validity'] is not None) if any(r['citations'] for r in rows) else None,
            'mean_exact_quote_validity':mean(r['exact_quote_validity'] for r in rows if r['exact_quote_validity'] is not None) if any(r['citations'] for r in rows) else None},
        'limitations':['Exact quote presence establishes traceability, not source truth or paraphrase entailment.',
            'Undefined citation/redundancy values are excluded, not treated as zero.',
            'No real-world accuracy or calibrated confidence is claimed. Human quality fields require independent annotations.'],
        'output':result}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.input.resolve()==args.output.resolve():parser.error('Output must not overwrite input')
    try:
        report=evaluate(json.loads(args.input.read_text(encoding='utf-8-sig')))
        write_json_atomic(args.output,report);print(json.dumps(report['aggregate'],indent=2));return 0
    except (ValueError,OSError,RuntimeError) as exc:print(str(exc));return 2


if __name__=='__main__':raise SystemExit(main())
