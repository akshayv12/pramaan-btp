"""Blind claim review export and per-language, question-disjoint calibration tools."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from .handoff import write_json_atomic
from .nli import MODEL,REVISION

SCHEMA='task5.human_review.v1'
IDENTITY=('schema_version','task_id','query_group','query','language','claim','references')


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def export_tasks(outputs):
    tasks=[];signals={};seen=set()
    for output in outputs:
        for query in output['task5_output']['queries']:
            result=query['result'];review=result.get('diagnostics',{}).get('review',{})
            nli=review.get('nli',{});consistency=review.get('consistency',{})
            accepted=nli.get('accepted_claim_indices',list(range(len(result['claims']))))
            passages={p['id']:p for p in result.get('selected_passages',result['used_passages'])}
            proposed=review.get('claim_decisions',[])
            candidates=proposed if proposed and all('text' in row and 'evidence' in row for row in proposed) else result['claims']
            for position,claim in enumerate(candidates):
                original_index=claim.get('claim_index',accepted[position] if position<len(accepted) else position)
                references=[{'passage_id':r['passage_id'],'quote':r['quote'],'passage':passages[r['passage_id']]['text'],
                             'url':passages[r['passage_id']].get('url_or_provenance')} for r in claim['evidence']]
                identity={'query':query['query'],'language':result['target_language'],'claim':claim['text'],'references':references}
                task_id=digest(identity)[:24]
                if task_id in seen:continue
                seen.add(task_id)
                tasks.append({'schema_version':SCHEMA,'task_id':task_id,'query_group':digest(query['query'].casefold().strip())[:20],
                              **identity,'review_status':'pending','reviewer_id':None,'human_review_attested':False,
                              'faithfulness':None,'relevant':None,'language_fidelity':None,'review_note':''})
                row=next((r for r in nli.get('claims',[]) if r['claim_index']==original_index),{})
                sampled=next((r for r in consistency.get('claims',[]) if r['claim_index']==original_index),{})
                decision=next((r for r in review.get('claim_decisions',[]) if r['claim_index']==original_index),{})
                signals[task_id]={'language':result['target_language'],'query_group':tasks[-1]['query_group'],
                                 'nli_entailment':row.get('entailment'),'nli_contradiction':row.get('contradiction'),
                                 'judge_supported':decision.get('supported'),'consistency':sampled.get('survival_fraction'),
                                 'accepted_by_pipeline':decision.get('accepted',True),
                                 'implementation_fingerprint':output['task5_output'].get('implementation_fingerprint'),
                                 'nli_model':nli.get('model'),'nli_revision':nli.get('revision')}
    return tasks,{'schema_version':'task5.review_signals.v1','signals':signals}


def reviewed_rows(tasks,reviews):
    originals={row['task_id']:row for row in tasks};accepted=[]
    if len(originals)!=len(tasks):raise ValueError('Duplicate review tasks')
    seen=set()
    for row in reviews:
        task_id=row.get('task_id')
        if task_id not in originals or task_id in seen:raise ValueError('Unknown or repeated review task ID')
        seen.add(task_id);original=originals[task_id]
        if any(row.get(key)!=original.get(key) for key in IDENTITY):raise ValueError('Review changed the original question, claim or references')
        if row.get('review_status')=='pending':continue
        reviewer=row.get('reviewer_id')
        if row.get('review_status')!='complete' or row.get('human_review_attested') is not True or not isinstance(reviewer,str) or reviewer.strip().lower() in {'','pending','todo','unknown'}:
            raise ValueError('Completed reviews need a named reviewer and explicit human-review attestation')
        if row.get('faithfulness') not in {'supported','unsupported','contradicted','unclear'} or type(row.get('relevant')) is not bool or row.get('language_fidelity') not in {'pass','partial','fail'}:
            raise ValueError('Complete the faithfulness, relevance and language judgments')
        accepted.append(row)
    return accepted


def metrics(pairs):
    tp=sum(y and p for y,p in pairs);tn=sum(not y and not p for y,p in pairs)
    fp=sum(not y and p for y,p in pairs);fn=sum(y and not p for y,p in pairs)
    sensitivity=tp/(tp+fn) if tp+fn else None;specificity=tn/(tn+fp) if tn+fp else None
    return {'items':len(pairs),'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'balanced_accuracy':(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':sensitivity}


def fit_threshold(rows,feature,positive):
    candidates=[]
    for threshold in (.5,.6,.7,.8,.85,.9,.95):
        pairs=[(positive(row),row[feature]>=threshold) for row in rows if row.get(feature) is not None]
        score=metrics(pairs)['balanced_accuracy']
        candidates.append((score if score is not None else -1,threshold))
    return max(candidates)[1]


def calibrate(tasks,reviews,signals):
    reviewed=reviewed_rows(tasks,reviews);by_language=defaultdict(list)
    for row in reviewed:
        if row['faithfulness']=='unclear':continue
        signal=signals.get(row['task_id'],{})
        if signal.get('nli_model')!=MODEL or signal.get('nli_revision')!=REVISION:continue
        if signal.get('language')!=row['language'] or signal.get('query_group')!=row['query_group']:raise ValueError('Review signal identity mismatch')
        for key in ('nli_entailment','nli_contradiction','consistency'):
            value=signal.get(key)
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=1):raise ValueError('Invalid review signal')
        by_language[row['language']].append({**row,**signal})
    languages={}
    for language in sorted({row['language'] for row in tasks}):
        rows=by_language[language]
        groups=sorted({row['query_group'] for row in rows},key=digest)
        train_groups=set(groups[::2]);train=[row for row in rows if row['query_group'] in train_groups]
        heldout=[row for row in rows if row['query_group'] not in train_groups]
        profile={'status':'insufficient_reviewed_data','human_review_attested':bool(rows),'reviewed_items':len(rows),
                 'train_items':len(train),'heldout_items':len(heldout),'question_groups':len(groups),
                 'train_question_groups':sorted(train_groups),'heldout_question_groups':sorted(set(groups)-train_groups),
                 'requirements':'At least 10 train and 10 held-out NLI-scored items per language, both faithful and unfaithful; disjoint question groups.'}
        nli_train=[row for row in train if row.get('nli_entailment') is not None and row.get('nli_contradiction') is not None]
        nli_heldout=[row for row in heldout if row.get('nli_entailment') is not None and row.get('nli_contradiction') is not None]
        if len(nli_train)>=10 and len(nli_heldout)>=10 and all(len({r['faithfulness']=='supported' for r in split})==2 for split in (nli_train,nli_heldout)):
            entail=fit_threshold(nli_train,'nli_entailment',lambda r:r['faithfulness']=='supported')
            contradiction_evaluable=all(len({r['faithfulness']=='contradicted' for r in split})==2 for split in (nli_train,nli_heldout))
            contradict=fit_threshold(nli_train,'nli_contradiction',lambda r:r['faithfulness']=='contradicted') if contradiction_evaluable else .8
            stability=fit_threshold(train,'consistency',lambda r:r['faithfulness']=='supported')
            features={}
            for name,feature,threshold in (('nli','nli_entailment',entail),('sampling_consistency','consistency',stability)):
                features[name]=metrics([(r['faithfulness']=='supported',r[feature]>=threshold) for r in heldout if r.get(feature) is not None])
            features['judge']=metrics([(r['faithfulness']=='supported',r['judge_supported']) for r in heldout if type(r.get('judge_supported')) is bool])
            features['all_three']=metrics([(r['faithfulness']=='supported',r['nli_entailment']>=entail and r['judge_supported'] and r['consistency']>=stability)
                                          for r in heldout if r.get('nli_entailment') is not None and type(r.get('judge_supported')) is bool and r.get('consistency') is not None])
            profile.update(status='reviewed_holdout_evaluated',train_items=len(nli_train),heldout_items=len(nli_heldout),
                           thresholds={'entailment':entail,'contradiction':contradict,'consistency':stability},heldout_metrics=features,
                           contradiction_threshold_status='reviewed_holdout_evaluated' if contradiction_evaluable else 'default_no_contradiction_examples',
                           note='Judge booleans are evaluated rather than fitted as continuous probabilities. Missing signals are excluded, never filled with invented scores.')
        languages[language]=profile
    return {'schema_version':'task5.calibration.v1','nli_model':MODEL,'nli_revision':REVISION,'languages':languages,
            'reviewed_items':len(reviewed),'pending_items':len(tasks)-len(reviewed),
            'limitations':['Human review is explicitly attested, not independently authenticated by this tool.',
                          'Held-out metrics describe this small set, not universal accuracy or truth probabilities.',
                          'Runtime calibration applies NLI thresholds only; judge/consistency metrics are reported separately.']}


def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    export=sub.add_parser('export');export.add_argument('outputs',type=Path,nargs='+');export.add_argument('--tasks',type=Path,required=True);export.add_argument('--signals',type=Path,required=True)
    fit=sub.add_parser('calibrate');fit.add_argument('--tasks',type=Path,required=True);fit.add_argument('--reviews',type=Path,required=True);fit.add_argument('--signals',type=Path,required=True);fit.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='export':
            if args.tasks.exists() or args.signals.exists():raise ValueError('Choose new task/signal files; existing reviews are never overwritten')
            tasks,signals=export_tasks([json.loads(path.read_text()) for path in args.outputs])
            args.tasks.parent.mkdir(parents=True,exist_ok=True)
            args.tasks.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in tasks))
            write_json_atomic(args.signals,signals);print(json.dumps({'tasks':len(tasks),'status':'pending_human_review'}))
        else:
            report=calibrate(read_jsonl(args.tasks),read_jsonl(args.reviews),json.loads(args.signals.read_text())['signals'])
            write_json_atomic(args.output,report);print(json.dumps({'reviewed_items':report['reviewed_items'],'pending_items':report['pending_items'],'languages':{k:v['status'] for k,v in report['languages'].items()}}))
        return 0
    except (OSError,ValueError,KeyError) as error:print(str(error));return 2


if __name__=='__main__':raise SystemExit(main())
