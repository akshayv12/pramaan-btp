"""Offline Task 5 ablations. Planning is free; --execute explicitly makes model calls."""
import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import time
from .config import Task5Config,load_env_file,implementation_fingerprint
from .handoff import summarize_task4_payload,write_json_atomic,validate_envelope


def plan(payload,ks,orders,strategies,max_queries=1,max_runs=12):
    rows=payload.get('task4_output',{}).get('queries',[payload])[:max_queries]
    for row in rows:validate_envelope(row)
    if not ks or any(type(k) is not int or not 1<=k<=12 for k in ks):raise ValueError('k must be between 1 and 12')
    if not orders or any(v not in {'relevance','reverse','shuffle'} for v in orders):raise ValueError('Unknown ordering')
    if not strategies or any(v not in {'direct','translate_then_summarize'} for v in strategies):raise ValueError('Unknown translation strategy')
    jobs=[{'query_index':i,'query':row['query'],'k':k,'order':order,'strategy':strategy}
          for i,row in enumerate(rows) for k,order,strategy in product(ks,orders,strategies)]
    if len(jobs)>max_runs:raise ValueError(f'This plan needs {len(jobs)} runs; reduce variants or explicitly raise --max-runs (maximum 24)')
    return rows,jobs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--provider',choices=['auto','gemini','ollama'],default='auto');parser.add_argument('--model')
    parser.add_argument('--language',choices=['en','hi']);parser.add_argument('--ks',default='2,4')
    parser.add_argument('--orders',default='relevance,reverse');parser.add_argument('--strategies',default='direct,translate_then_summarize')
    parser.add_argument('--max-queries',type=int,default=1);parser.add_argument('--max-runs',type=int,default=12)
    parser.add_argument('--verification',choices=['standard','thorough'],default='standard');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.max_queries<=10 or not 1<=args.max_runs<=24:parser.error('Use 1–10 queries and 1–24 runs')
    load_env_file(Path(__file__).resolve().parents[1]/'.env')
    try:
        payload=json.loads(args.input.read_text(encoding='utf-8-sig'))
        rows,jobs=plan(payload,[int(v) for v in args.ks.split(',')],args.orders.split(','),args.strategies.split(','),args.max_queries,args.max_runs)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        report={'schema_version':'task5.experiments.v1','status':'planned','input':str(args.input.resolve()),
                'fingerprint':implementation_fingerprint(),'variants':jobs,'results':[],
                'limits':['Original Task 4 ranks and scores are unchanged.','Automated checks are not human semantic-quality judgments.',
                          'No best k/order/translation strategy is selected without annotated quality results.']}
        destination=args.output_dir/'experiments.json'
        write_json_atomic(destination,report)
        if args.execute:
            base=Task5Config.from_env(provider=args.provider,model=args.model,min_query_rel=.10,allow_fallback=False,verification_profile=args.verification)
            for index,job in enumerate(jobs,1):
                started=time.perf_counter();record={'variant':job,'status':'failed'}
                try:
                    config=replace(base,max_passages=job['k'],passage_order=job['order'],cross_lingual_strategy=job['strategy'])
                    result=summarize_task4_payload(rows[job['query_index']],config=config,target_language=args.language)
                    output=args.output_dir/f'run-{index:02}.json';write_json_atomic(output,result)
                    answer=result['task5_output']['queries'][0]['result'];review=answer['diagnostics']['review']
                    record.update(status='completed',output=output.name,selected_passages=len(answer['selected_passages']),
                                  accepted_claims=len(answer['claims']),answer_status=answer['answer_status'],reliability=answer['reliability'],
                                  generation_input=review.get('generation_input'),nli=review.get('nli'),criteria=review.get('criteria'),
                                  human_faithfulness=None,human_relevance=None,human_readability=None)
                except (ValueError,RuntimeError,OSError) as error:record['error']=str(error)[:300]
                record['duration_seconds']=round(time.perf_counter()-started,3);report['results'].append(record)
                write_json_atomic(destination,report)
                print(f"Run {index}/{len(jobs)}: {record['status']}",flush=True)
                if any(code in record.get('error','') for code in ('HTTP 401','HTTP 403','HTTP 404','HTTP 429')):
                    report['stopped_reason']='Model access/quota failed; remaining variants were not run.';break
            report['status']='completed' if len(report['results'])==len(jobs) and all(r['status']=='completed' for r in report['results']) else 'incomplete'
            write_json_atomic(destination,report)
        print(str(destination))
        return 0 if report['status'] in {'planned','completed'} else 2
    except (ValueError,OSError,RuntimeError) as error:print(str(error));return 2


if __name__=='__main__':raise SystemExit(main())
