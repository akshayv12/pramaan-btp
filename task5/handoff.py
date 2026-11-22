"""Task 5 adapter for the existing, unchanged Task 4 single-query and batch formats."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from .config import Task5Config,load_env_file,IMPLEMENTATION_VERSION,implementation_fingerprint
from .pipeline import Task5Pipeline
from .schemas import EvidencePassage,unit_score
from .text import detect_language

SOURCE_TIERS={'academic':'High','institutional':'High','government':'High','scholarly':'High',
              'book':'Medium','manuscript':'Medium','news':'Medium','blog':'Low','anonymous':'Low'}
LANGUAGES={'hindi':'hi','devanagari':'hi','hindi_devanagari':'hi','roman_hindi':'hi','english':'en','sanskrit':'sa','hi':'hi','en':'en','sa':'sa'}


def _object(value,label):
    if not isinstance(value,dict):raise ValueError(label+' must be an object')
    return value


def validate_envelope(package):
    _object(package,'Task 4 envelope')
    if package.get('schema_version')!='1.0.0':raise ValueError('Expected Task 4 envelope schema 1.0.0')
    query=package.get('query')
    if not isinstance(query,str) or not 1<=len(query.strip())<=2000:raise ValueError('Invalid Task 4 query')
    rows=package.get('ranked_evidence')
    if not isinstance(rows,list) or len(rows)>500:raise ValueError('Invalid ranked_evidence')
    if package.get('status')!=('ok' if rows else 'no_relevant_evidence'):raise ValueError('Task 4 evidence/status mismatch')
    seen=set()
    for position,row in enumerate(rows,1):
        _object(row,'ranked evidence')
        pid=row.get('passage_id')
        if not isinstance(pid,str) or not pid.strip() or pid in seen:raise ValueError('Invalid/duplicate upstream passage ID')
        seen.add(pid)
        if type(row.get('rank')) is not int or row['rank']!=position:raise ValueError('Task 4 ranks must be consecutive')
        if not isinstance(row.get('text'),str) or not row['text'].strip() or len(row['text'])>100000:raise ValueError('Invalid passage text')
        for name in ('query_relevance','indian_context_relevance','source_signal','final_score'):unit_score(name,row.get(name))
        provenance=_object(row.get('provenance',{}),'passage provenance')
        for key in ('source_metadata','extraction_metadata'):_object(provenance.get(key,{}),key)
        for key in ('feature_breakdown','source_support'):
            if key in row:_object(row[key],key)
    trace=_object(package.get('traceability'),'Task 4 traceability')
    if trace.get('selected_evidence_ids')!=[r['passage_id'] for r in rows]:raise ValueError('Traceability passage IDs do not match evidence')
    references=trace.get('selected_references')
    if not isinstance(references,list) or len(references)!=len(rows):raise ValueError('Traceability must cover every passage')
    for ref,row in zip(references,rows):
        if not isinstance(ref,dict) or ref.get('passage_id')!=row['passage_id'] or type(ref.get('reference_available')) is not bool:raise ValueError('Invalid reference record')
        provenance=row.get('provenance',{});fields={**provenance.get('source_metadata',{}),**provenance.get('extraction_metadata',{})}
        for key in ('document_id','source_id','url','title','language','source_group_id','origin_document_id','method','page_number','ocr_confidence','image_id','page_id'):
            if key in ref and key in fields and ref[key]!=fields[key]:raise ValueError('Conflicting traceability alias: '+key)
    expected_available=bool(rows) and all(ref['reference_available'] for ref in references)
    if trace.get('references_available') is not expected_available:raise ValueError('Inconsistent reference availability')
    if trace.get('source_independence_established') is not False or trace.get('text_similarity_establishes_agreement') is not False:
        raise ValueError('Task 4 grouping/similarity cannot establish independence or agreement')
    scope='all_input_candidates_before_filtering' if any('source_support' in row for row in rows) else 'not_included'
    if trace.get('source_support_scope')!=scope:raise ValueError('Inconsistent support-counter scope')
    provenance=_object(package.get('provenance'),'envelope provenance')
    if 'document_status_counts' in provenance:_object(provenance['document_status_counts'],'document status counts')
    if provenance.get('kind') not in {'task3_output','constructed_example'} or not isinstance(provenance.get('description'),str) or not provenance['description'].strip():
        raise ValueError('Invalid Task 4 evidence provenance')
    if type(package.get('is_simulated')) is not bool or package['is_simulated']!=(provenance['kind']=='constructed_example'):raise ValueError('Inconsistent simulation flag')
    return package


def adapt_task4_envelope(package):
    validate_envelope(package)
    passages=[];mapping={};markers=[]
    for row,ref in zip(package['ranked_evidence'],package['traceability']['selected_references']):
        provenance=row.get('provenance',{});source={**ref,**provenance.get('source_metadata',{})};extraction={**ref,**provenance.get('extraction_metadata',{})}
        method=extraction.get('method') or 'direct'
        if not isinstance(method,str):raise ValueError('Extraction method must be a string')
        ocr='ocr' in method.lower()
        confidence=extraction.get('ocr_confidence') if ocr else None
        if confidence is not None:unit_score('already-normalized OCR confidence',confidence)
        kind=source.get('source_type') or source.get('task2_source_type') or 'unknown'
        if not isinstance(kind,str):raise ValueError('Source type must be a string')
        lang=LANGUAGES.get(str(source.get('language','')).lower()) or LANGUAGES.get(str(extraction.get('detected_language','')).lower()) or detect_language(row['text'])
        pid='S'+str(row['rank'])
        metadata={'upstream_passage_id':row['passage_id'],'upstream_document_id':source.get('document_id'),
            'task4':copy.deepcopy({k:v for k,v in row.items() if k!='text'}),'upstream_reference':copy.deepcopy(ref),
            'source_metadata':copy.deepcopy(provenance.get('source_metadata',{})),
            'extraction_metadata':copy.deepcopy(provenance.get('extraction_metadata',{})),
            'independence_verified':False,'synthetic':package['is_simulated'],
            'authority_basis':'declared source category only; numeric Task 4 aggregate is preserved, not reinterpreted'}
        p=EvidencePassage(pid,row['text'],kind,row['query_relevance'],row['indian_context_relevance'],SOURCE_TIERS.get(kind.casefold(),'Unknown'),
            confidence,lang,source.get('canonical_url') or source.get('url'),source.get('title'),source.get('source_id'),source.get('organization'),extraction.get('page_number'),method,metadata)
        passages.append(p);mapping[pid]=row['passage_id']
        keys=set();identity=p.source_identity()
        if identity:keys.add(identity)
        # Include domain even when a publisher/source ID is supplied.
        domain=EvidencePassage(pid,p.text,kind,p.query_rel,p.india_rel,p.source_signal,url_or_provenance=p.url_or_provenance).source_identity()
        if domain:keys.add(domain)
        if source.get('document_id'):keys.add('document:'+str(source['document_id']))
        for key in ('origin_document_id','copied_from_document_id','source_group_id'):
            value=source.get(key) or row.get('source_support',{}).get(key)
            if value:keys.add(('group:' if key=='source_group_id' else 'document:')+str(value))
        markers.append(keys)
    groups=[]
    for index,keys in enumerate(markers):
        if not keys:continue
        matching=[g for g in groups if g[0]&keys];merged=set(keys);indices={index}
        for group in matching:merged.update(group[0]);indices.update(group[1]);groups.remove(group)
        groups.append((merged,indices))
    for keys,indices in groups:
        if all(key.startswith('document:') for key in keys):continue
        identity=hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()[:16]
        for index in indices:passages[index].metadata['task5_source_group']=identity
    return passages,mapping


def summarize_task4_payload(payload,*,config=None,target_language=None):
    _object(payload,'Task 4 payload');config=config or Task5Config.from_env(min_query_rel=.10)
    batch=payload.get('task4_output')
    if batch is not None:
        _object(batch,'task4_output')
        if batch.get('schema_version')!='1.0' or not isinstance(batch.get('queries'),list):raise ValueError('Expected Task 4 batch schema 1.0')
        envelopes=batch['queries']
    else:batch={};envelopes=[payload]
    if len(envelopes)>200:raise ValueError('At most 200 queries per batch')
    for envelope in envelopes:validate_envelope(envelope)
    output=[]
    for envelope in envelopes:
        evidence,mapping=adapt_task4_envelope(envelope)
        ranking=batch.get('ranking') or {}
        rank_config=ranking.get('config',{}) if isinstance(ranking,dict) else {}
        weights=rank_config.get('relevance_weights',{}) if isinstance(rank_config,dict) else {}
        if not isinstance(weights,dict):weights={}
        retrieval_mode='semantic' if weights.get('semantic_weight',0)>0 else 'lexical' if weights.get('lexical_weight',0)>0 else 'unknown'
        for passage in evidence:passage.metadata['task4_retrieval_mode']=retrieval_mode
        result=Task5Pipeline(config=config).run(envelope['query'],evidence,target_language)
        result.warnings.append('Upstream source-support counters are not independent corroboration of the answer.')
        if envelope['provenance'].get('document_status_counts',{}).get('failed'):
            result.warnings.append('Some upstream documents failed extraction; coverage may be incomplete.')
        output.append({'query':envelope['query'],'status':'answered' if result.claims else 'insufficient_evidence',
            'answer_status':result.answer_status,'task4_status':envelope['status'],'is_simulated':envelope['is_simulated'],
            'citation_to_task4_passage_id':mapping,'upstream_provenance':copy.deepcopy(envelope['provenance']),
            'upstream_traceability':copy.deepcopy(envelope['traceability']),
            'integration':{'task5':'implemented_consumer','actual_task5_available':True},
            'public_output':result.to_public_dict(),'display':result.to_formatted_display(),'result':result.to_dict()})
    return {'task5_output':{'schema_version':'1.0','implementation_version':IMPLEMENTATION_VERSION,'execution_mode':'resume_from_task4',
        'implementation_fingerprint':implementation_fingerprint(),
        'metadata':copy.deepcopy(batch.get('metadata',{})),'task3_summary':copy.deepcopy(batch.get('task3_summary')),
        'task4_ranking':copy.deepcopy(batch.get('ranking')),'queries':output,
        'summary':{'questions':len(output),'answered':sum(o['status']=='answered' for o in output),'abstained':sum(o['status']!='answered' for o in output)}}}


def _atomic(path,value,text=False):
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(value.rstrip()+'\n' if text else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        os.replace(temporary,path)
    finally:
        if temporary:temporary.unlink(missing_ok=True)


def write_json_atomic(path,value):_atomic(path,value)

def process_task4_output(source,output=None,*,config=None,target_language=None):
    source=Path(source).resolve();destination=Path(output).resolve() if output else source.with_name('task5_output.json')
    if source==destination:raise ValueError('Task 5 output must differ from its Task 4 input')
    value=summarize_task4_payload(json.loads(source.read_text(encoding='utf-8-sig')),config=config,target_language=target_language)
    write_json_atomic(destination,value);return value


def main(argv=None):
    parser=argparse.ArgumentParser(description='Task 4 ranked evidence -> cited answer, supporting sources and support label')
    parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--text-output',type=Path)
    parser.add_argument('--provider',choices=['auto','extractive','gemini','ollama']);parser.add_argument('--model')
    parser.add_argument('--language',choices=['hi','en']);parser.add_argument('--strict',action='store_true')
    parser.add_argument('--min-query-rel',type=float,default=.10)
    args=parser.parse_args(argv);load_env_file(Path(__file__).resolve().parents[1]/'.env');load_env_file()
    try:
        destination=args.output or args.input.with_name('task5_output.json')
        if args.text_output and args.text_output.resolve() in {args.input.resolve(),destination.resolve()}:raise ValueError('Text output must differ from input and JSON output')
        config=Task5Config.from_env(provider=args.provider,model=args.model,min_query_rel=args.min_query_rel,allow_fallback=not args.strict)
        value=process_task4_output(args.input,args.output,config=config,target_language=args.language)
        displays=[o['display'] for o in value['task5_output']['queries']]
        if args.text_output:_atomic(args.text_output,'\n\n'.join(displays),text=True)
        print('\n\n'.join(displays));return 0
    except (ValueError,TypeError,OSError,RuntimeError) as exc:
        print('[TASK 5] '+type(exc).__name__+': '+str(exc),file=sys.stderr);return 2
