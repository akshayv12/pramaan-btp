"""A temporary query-specific document collection inspired by Open-MDS.

Open-MDS indexing.py exposes docno/text corpus records. Its canonical MDS
format joins documents with a separator. Here Task 4 already performed
retrieval/ranking, so no second retriever, persistent index or Java is loaded.
"""
import hashlib
import json

REFERENCE_COMMIT='c95c3995ff93bea10c3166ff7f587b1e0a34c568'


def build_collection(query,passages):
    documents=[];groups={}
    for p in passages:
        key=p.metadata.get('upstream_document_id') or p.url_or_provenance or 'passage:'+p.id
        if key not in groups:
            doc={'docno':'D'+str(len(documents)+1),'passages':[]}
            documents.append(doc);groups[key]=doc
        groups[key]['passages'].append({'id':p.id,'text':p.text,'language':p.language,'page':p.page})
    identity=hashlib.sha256(json.dumps({'query':query,'documents':documents},sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]
    return {'collection_id':identity,'query':query,'documents':documents,'scope':'one query; no persistent index',
            'reference_commit':REFERENCE_COMMIT}


def export_open_mds_record(collection,reference=None,separator='|||||'):
    if not isinstance(separator,str) or not separator.strip():raise ValueError('A document separator is required')
    texts=['\n'.join(p['text'] for p in d['passages']) for d in collection['documents']]
    if any(separator in text for text in texts):raise ValueError('Separator occurs inside evidence')
    return {'document':(' '+separator+' ').join(texts),'summary':reference if reference is not None else '',
            'query':collection['query'],'reference_available':reference is not None,
            'document_ids':[d['docno'] for d in collection['documents']],
            'passage_ids':[[p['id'] for p in d['passages']] for d in collection['documents']]}
