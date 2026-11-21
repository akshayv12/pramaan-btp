"""Requirement tests for the replacement Task 5, independent of the deleted runtime."""
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from task5 import Task5Config,Task5Pipeline,EvidencePassage,GroundedClaim,ClaimEvidence,summarize_with_reliability
from task5.collection import build_collection,export_open_mds_record
from task5.evidence import select_evidence,detect_conflicts
from task5.generation import validate_claims
from task5.providers import ProviderError,StructuredProvider,post_json
from task5.reliability import assess
from task5.text import normalize,detect_language
from test_handoff import envelope
from task5.handoff import main,summarize_task4_payload


def passage(id='S1',text='The archive provides public catalogue access.',**values):
    data=dict(id=id,text=text,source_type='institutional',query_rel=.95,india_rel=.8,source_signal='High',language='en',url_or_provenance='https://archive.one.org/x')
    data.update(values);return EvidencePassage(**data)


class ModelStub:
    name='gemini'
    def __init__(self,*responses):self.responses=iter(responses);self.prompts=[]
    def generate(self,prompt,schema):
        self.prompts.append(prompt);response=next(self.responses)
        if isinstance(response,Exception):raise response
        return response


def draft(p,text=None,quote=None,pid=None):
    return {'claims':[{'text':text or p.text,'evidence':[{'passage_id':pid or p.id,'quote':quote or p.text}]}]}


class RequirementsTests(unittest.TestCase):
    def setUp(self):self.config=Task5Config(provider='extractive')

    def test_public_output_matches_specification_without_percentage(self):
        r=summarize_with_reliability('What access does the archive provide?',[passage()],config=self.config)
        public=r.to_public_dict();display=r.to_formatted_display()
        self.assertTrue(public['answer']);self.assertEqual(public['sources_used'],['S1'])
        self.assertIn(public['reliability'],['HIGH','MEDIUM','LOW']);self.assertTrue(public['why'])
        self.assertIn('Answer:',display)
        for label in ('Reliability:','Why:','Sources used:'):self.assertIn(label,display)
        self.assertNotIn('%',display);self.assertNotIn('Generation:',display)
        self.assertEqual(public['supporting_sources'][0]['quotes'],[passage().text])

    def test_claims_are_concise_with_exact_source_quotes(self):
        a=passage();b=passage('S2','The library maintains a manuscript catalogue. Readers request research appointments.',url_or_provenance='https://library.other.org/y')
        r=Task5Pipeline(self.config).run('How can researchers access catalogue records?',[a,b])
        self.assertLessEqual(len(r.claims),3);self.assertLessEqual(sum(len(c.text) for c in r.claims),1100)
        by_id={p.id:p for p in r.used_passages}
        for c in r.claims:
            for e in c.evidence:
                self.assertIn(normalize(c.text),normalize(e.quote));self.assertIn(normalize(e.quote),normalize(by_id[e.passage_id].text))
        self.assertGreaterEqual(len(r.cited_sources),2)

    def test_source_copies_and_one_publisher_cannot_dominate(self):
        items=[passage(str(i),text=t) for i,t in enumerate(['The archive provides reading rooms.','The archive preserves manuscripts and photographs.','The archive opens for visitors at noon.'])]
        items.append(passage('mirror',text=items[0].text,url_or_provenance='https://copy.other.org'))
        selected,pool,decisions=select_evidence('What does the archive provide?',items,self.config)
        self.assertLessEqual(len(selected),2);self.assertTrue(any('copy' in d['reason'] for d in decisions))

    def test_number_and_negation_conflicts_survive_deduplication(self):
        for text in ('The city archive opened to the public in 1990.','The city archive did not open to the public in 1985.'):
            a=passage(text='The city archive opened to the public in 1985.');b=passage('S2',text,url_or_provenance='https://other.org')
            r=Task5Pipeline(self.config).run('When did the city archive open?',[a,b])
            self.assertEqual(len(r.selected_passages),2);self.assertTrue(r.conflicts);self.assertNotEqual(r.reliability.value,'HIGH')

    def test_indian_context_never_increases_the_reliability_label(self):
        p=passage(india_rel=0);a=Task5Pipeline(self.config).run('How is archive access provided?',[p]);b=Task5Pipeline(self.config).run('How is archive access provided?',[replace(p,india_rel=1)])
        self.assertEqual(a.reliability,b.reliability);self.assertFalse(b.factors['context_increases_trust'])
        self.assertEqual(a.factors['source_quality'],b.factors['source_quality'])

    def test_high_gate_requires_documented_support_not_only_many_sources(self):
        a=passage(metadata={'independence_verified':True});b=passage('S2',a.text+' Staff preserve maps and manuscripts.',url_or_provenance='https://other.org',metadata={'independence_verified':True})
        claim=GroundedClaim(a.text,[ClaimEvidence(a.id,a.text),ClaimEvidence(b.id,a.text)])
        rating,_,_=assess([claim],[a,b],[a,b],[],{'status':'skipped'})
        self.assertEqual(rating.value,'HIGH')
        b.metadata['independence_verified']=False
        self.assertNotEqual(assess([claim],[a,b],[a,b],[],{'status':'skipped'})[0].value,'HIGH')

    def test_weak_sources_empty_and_bad_ocr_have_low_support(self):
        for items in ([],[passage(source_signal='Low')],[passage(query_rel=.1)],[passage(ocr_conf=.2,extraction_method='ocr')]):
            r=Task5Pipeline(self.config).run('What access exists?',items)
            self.assertEqual(r.reliability.value,'LOW')
        p=passage(ocr_conf=.63,extraction_method='ocr');before=p.to_dict()
        r=Task5Pipeline(self.config).run('What access exists?',[p]);self.assertEqual(p.to_dict(),before)
        self.assertEqual(r.used_passages[0].ocr_conf,.63);self.assertEqual(r.reliability.value,'LOW')

    def test_hindi_and_roman_hindi_routing(self):
        p=passage(text='संग्रहालय में पांडुलिपियों की सार्वजनिक सूची उपलब्ध है।',language='hi')
        for query in ('संग्रहालय में क्या उपलब्ध है?','Sangrahalaya mein kya milta hain?'):
            r=Task5Pipeline(self.config).run(query,[p]);self.assertEqual(r.target_language,'hi');self.assertIn('उत्तर:',r.to_formatted_display())
        self.assertEqual(detect_language('How does the engine work?'),'en')

    def test_extracting_other_language_does_not_pretend_to_translate(self):
        r=Task5Pipeline(self.config).run('क्या उपलब्ध है?',[passage()],target_language='hi')
        self.assertEqual(r.answer_status,'limited');self.assertTrue(any('does not translate' in w for w in r.warnings))

    def test_only_two_model_calls_for_draft_and_combined_review(self):
        p=passage();provider=ModelStub(draft(p),{'supported':[True],'conflicts':[]})
        r=Task5Pipeline(Task5Config(provider='gemini',allow_fallback=False),provider=provider).run('What access exists?',[p])
        self.assertEqual(len(provider.prompts),2);self.assertEqual(r.generation_mode,'gemini')
        self.assertEqual(r.diagnostics['review']['status'],'completed');self.assertNotEqual(r.reliability.value,'HIGH')

    def test_invalid_ids_quotes_review_and_unsupported_claims_fail_closed(self):
        p=passage();cfg=Task5Config(provider='gemini',allow_fallback=False)
        for response in (draft(p,pid='Unknown'),draft(p,quote='Invented source quotation.')):
            with self.assertRaises(ProviderError):Task5Pipeline(cfg,provider=ModelStub(response)).run('What exists?',[p])
        for response in ({'supported':['true'],'conflicts':[]},{'supported':[False],'conflicts':[]},{'supported':[True]}):
            with self.assertRaises(ProviderError):Task5Pipeline(cfg,provider=ModelStub(draft(p,text='The archive grants everyone free entry.'),response)).run('What exists?',[p])

    def test_combined_model_conflicts_require_real_source_quotes(self):
        a=passage();b=passage('S2','Public catalogue access is not offered.',url_or_provenance='https://other.org')
        conflict={'left_id':a.id,'right_id':b.id,'left_quote':a.text,'right_quote':b.text,'reason':'Opposite access statements'}
        provider=ModelStub(draft(a),{'supported':[True],'conflicts':[conflict]})
        r=Task5Pipeline(Task5Config(provider='gemini'),provider=provider).run('What access exists?',[a,b])
        self.assertTrue(r.conflicts);self.assertEqual(r.conflicts[0]['method'],'model_review')
        conflict['right_quote']='A fabricated quotation.'
        with self.assertRaises(ProviderError):Task5Pipeline(Task5Config(provider='gemini',allow_fallback=False),provider=ModelStub(draft(a),{'supported':[True],'conflicts':[conflict]})).run('q',[a,b])

    def test_fallback_is_visible_and_does_not_claim_model_review(self):
        p=passage();r=Task5Pipeline(Task5Config(provider='gemini'),provider=ModelStub(ProviderError('timeout'))).run('What exists?',[p])
        self.assertEqual(r.generation_mode,'extractive_fallback');self.assertEqual(r.diagnostics['review']['status'],'failed')
        self.assertIn('limited lexical checks',r.reliability_explanation)

    def test_religious_views_are_not_automatically_contradictions(self):
        a=passage(text='A devotional tradition regards Rama as a divine incarnation.');b=passage('S2','A literary reading describes Rama as an ethical ideal.')
        self.assertEqual(detect_conflicts([a,b]),[])

    def test_query_specific_collection_has_no_cross_query_state(self):
        a=build_collection('First question?',[passage()]);b=build_collection('Second question?',[passage('Other')])
        self.assertNotEqual(a['collection_id'],b['collection_id']);record=export_open_mds_record(a)
        self.assertEqual(record['query'],'First question?');self.assertEqual(record['summary'],'');self.assertFalse(record['reference_available'])
        p=Task5Pipeline(self.config);first=p.run('First?',[passage('A')]);second=p.run('Second?',[passage('B')])
        self.assertNotIn('A',second.cited_sources);self.assertEqual(first.cited_sources,['A'])

    def test_no_heavy_dependencies_loaded_by_default(self):
        subprocess.run([sys.executable,'-c',"import sys,task5; assert not any(k in sys.modules for k in ['torch','transformers','bert_score','open_mds'])"],check=True,capture_output=True)

    def test_nonfinite_scores_and_duplicate_ids_are_rejected(self):
        for value in (math.nan,math.inf,-.1,1.1,True):
            with self.assertRaises(ValueError):passage(query_rel=value)
        with self.assertRaises(ValueError):Task5Pipeline(self.config).run('q',[passage(),passage()])
        with self.assertRaises(ValueError):Task5Pipeline(self.config).run('',[])

    def test_constructed_hindi_task4_replay_and_text_output(self):
        source=Path(__file__).parents[1]/'examples/supplied_task4_output.json';before=source.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'answer.json';text=Path(tmp)/'answer.txt'
            with patch('sys.stdout',io.StringIO()):code=main([str(source),'--provider','extractive','--output',str(out),'--text-output',str(text)])
            self.assertEqual(code,0);row=json.loads(out.read_text())['task5_output']['queries'][0]
            self.assertEqual(row['result']['reliability'],'LOW');self.assertEqual(row['answer_status'],'answered')
            self.assertEqual(text.read_text().strip(),row['display']);self.assertIn('Sources used:',text.read_text())
        self.assertEqual(source.read_bytes(),before)

    def test_transient_retry_and_authentication_failure(self):
        cfg=Task5Config(provider='gemini')
        with patch('task5.providers.urlopen',side_effect=[HTTPError('https://test',503,'busy',{},io.BytesIO()),io.BytesIO(b'{}')]) as call, patch('task5.providers.time.sleep') as sleep:
            self.assertEqual(post_json('https://test',{}, {},cfg),{});self.assertEqual(call.call_count,2);sleep.assert_called_once()
        with patch('task5.providers.urlopen',side_effect=HTTPError('https://test',401,'no',{},io.BytesIO())) as call:
            with self.assertRaises(ProviderError):post_json('https://test',{}, {},cfg)
            self.assertEqual(call.call_count,1)

    def test_model_can_answer_in_hindi_using_english_source_quotes(self):
        p=passage();text='अभिलेखागार की सार्वजनिक सूची उपलब्ध है।'
        provider=ModelStub(draft(p,text=text),{'supported':[True],'conflicts':[]})
        r=Task5Pipeline(Task5Config(provider='gemini',allow_fallback=False),provider=provider).run('क्या उपलब्ध है?',[p],target_language='hi')
        self.assertEqual(r.claims[0].text,text);self.assertEqual(r.claims[0].evidence[0].quote,p.text)
        self.assertTrue(r.to_formatted_display().startswith('उत्तर:'))

    def test_evaluation_does_not_invent_human_accuracy(self):
        from task5.evaluation import evaluate
        report=evaluate(envelope())
        self.assertEqual(report['aggregate']['mean_exact_quote_validity'],1)
        for key in ('human_coverage','human_relevance','human_readability','semantic_factual_support','human_reliability_explanation'):
            self.assertIsNone(report['cases'][0][key])

    def test_text_output_cannot_overwrite_task4_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'task4.json';source.write_text(json.dumps(envelope()));before=source.read_bytes()
            with patch('sys.stdout',io.StringIO()),patch('sys.stderr',io.StringIO()):
                code=main([str(source),'--provider','extractive','--text-output',str(source)])
            self.assertEqual(code,2);self.assertEqual(source.read_bytes(),before)

    def test_provider_records_version_without_serializing_key_or_thoughts(self):
        cfg=Task5Config(provider='gemini',model='available-model')
        response={'modelVersion':'resolved-version','candidates':[{'finishReason':'STOP','content':{'parts':[{'thought':True,'text':'private reasoning'},{'text':'{\"claims\":[]}'}]}}]}
        with patch.dict('os.environ',{'GEMINI_API_KEY':'DO_NOT_SERIALIZE'}),patch('task5.providers.post_json',return_value=response):
            provider=StructuredProvider(cfg);self.assertEqual(provider.generate('prompt',{}),{'claims':[]})
        self.assertEqual(provider.calls[0]['resolved_model'],'resolved-version')
        self.assertNotIn('DO_NOT_SERIALIZE',json.dumps(provider.calls));self.assertNotIn('private reasoning',json.dumps(provider.calls))


if __name__=='__main__':unittest.main()
