"""Local model transport and grounded paraphrasing contracts, without downloading weights."""
import json
import unittest
from unittest.mock import patch

from task5 import EvidencePassage,Task5Config,Task5Pipeline
from task5.generation import bind_evidence_cards,evidence_cards,qualification_flags,validate_claims,date_anchors,preserves_numbers_and_eras
from task5.providers import ProviderError,StructuredProvider


def passage(text='The archive provides public catalogue access.'):
    return EvidencePassage('S1',text,'institutional',.95,.6,'High',language='en',url_or_provenance='https://archive.example.org')


class LocalStub:
    name='ollama'
    def __init__(self,*responses):self.responses=iter(responses);self.prompts=[]
    def generate(self,prompt,schema):self.prompts.append((prompt,schema));return next(self.responses)


class LocalModelTests(unittest.TestCase):
    def setUp(self):self.config=Task5Config(provider='ollama',model='local-test',allow_fallback=False)

    def test_local_cli_defaults_are_bounded_for_memory_and_time(self):
        with patch.dict('os.environ',{},clear=True):config=Task5Config.from_env(provider='ollama')
        self.assertEqual(config.model_timeout,90)
        self.assertEqual(config.request_deadline,120)
        self.assertEqual(config.max_passages,4)
        self.assertEqual(config.context_chars,9000)
        self.assertEqual(config.ollama_num_ctx,8192)

    def test_provider_sends_schema_and_bounded_generation_options(self):
        schema={'type':'object','properties':{'ready':{'type':'boolean'}},'required':['ready']}
        response={'done':True,'done_reason':'stop','model':'local-test','message':{'content':'{"ready":true}','thinking':'not for serialization'},'eval_count':9}
        with patch.dict('os.environ',{'OLLAMA_BASE_URL':'http://127.0.0.1:11434'},clear=True),patch('task5.providers.post_json',return_value=response) as call:
            provider=StructuredProvider(self.config);self.assertEqual(provider.generate('Test',schema),{'ready':True})
        payload=call.call_args.args[1]
        self.assertEqual(payload['format'],schema)
        self.assertEqual(payload['options'],{'temperature':0,'num_ctx':8192,'num_predict':1024})
        self.assertEqual(payload['keep_alive'],'5m')
        self.assertIn('schema',payload['messages'][-1]['content'])
        self.assertEqual(provider.calls[0]['resolved_model'],'local-test')
        self.assertNotIn('not for serialization',json.dumps(provider.calls))

    def test_nonlocal_or_credentialed_endpoints_are_rejected(self):
        for endpoint in ('https://example.org','http://example.org','http://name:secret@localhost:11434'):
            with self.subTest(endpoint=endpoint),patch.dict('os.environ',{'OLLAMA_BASE_URL':endpoint},clear=True),self.assertRaises(ProviderError):
                StructuredProvider(self.config)

    def test_truncated_model_output_never_counts_as_completed(self):
        with patch.dict('os.environ',{},clear=True),patch('task5.providers.post_json',return_value={'done':True,'done_reason':'length','message':{'content':'{}'}}):
            provider=StructuredProvider(self.config)
            with self.assertRaises(ProviderError):provider.generate('Test',{})
            self.assertEqual(provider.calls[0]['status'],'failed')

    def test_model_phrases_answer_but_server_binds_the_exact_quote(self):
        p=passage();text='Researchers can access the archive’s public catalogue.'
        provider=LocalStub({'claims':[{'text':text,'evidence_ids':['E1']}]},{'supported':[True],'conflicts':[]})
        result=Task5Pipeline(self.config,provider=provider).run('What access does the archive provide?',[p])
        self.assertEqual(result.generation_mode,'ollama')
        self.assertEqual(result.claims[0].text,text)
        self.assertEqual(result.claims[0].evidence[0].quote,p.text)
        self.assertEqual(result.diagnostics['review']['evidence_binding'],'server_selected_verbatim_quotes')
        self.assertEqual(len(provider.prompts),2)
        schema=provider.prompts[0][1]
        self.assertEqual(schema['properties']['claims']['maxItems'],3)
        self.assertEqual(schema['properties']['claims']['items']['properties']['evidence_ids']['items']['enum'],['E1'])
        self.assertIn('English only',provider.prompts[0][0])
        self.assertNotIn('epic means महाकाव्य',provider.prompts[0][0])
        self.assertIn('English',schema['properties']['claims']['items']['properties']['text']['description'])

    def test_unknown_card_ids_cannot_fabricate_a_citation(self):
        provider=LocalStub({'claims':[{'text':'The archive offers free entry.','evidence_ids':['E99']}]})
        with self.assertRaises(ProviderError):Task5Pipeline(self.config,provider=provider).run('What access exists?',[passage()])

    def test_a_local_claim_cannot_conflate_two_evidence_cards(self):
        p=passage('The archive offers catalogue access. Funding is limited. The archive preserves maps.')
        cards=[{'id':'E1','passage_id':'S1','quote':'The archive offers catalogue access.'},
               {'id':'E2','passage_id':'S1','quote':'The archive preserves maps.'}]
        with self.assertRaises(ProviderError):
            bind_evidence_cards({'claims':[{'text':'The archive offers catalogue access and preserves maps.','evidence_ids':['E1','E2']}]},cards,[p],self.config)

    def test_overview_cards_exclude_details_about_the_pages_translation(self):
        p=passage('The Mahabharata is a Sanskrit epic about the Kuru family. This translation includes numerous books of the Mahabharata.')
        overview=evidence_cards('What do you know about Mahabharata?',[p],self.config)
        self.assertTrue(overview)
        self.assertFalse(any('This translation' in card['quote'] for card in overview))
        specific=evidence_cards('What does this Mahabharata translation include?',[p],self.config)
        self.assertTrue(any('This translation' in card['quote'] for card in specific))

    def test_lost_qualifier_is_omitted_before_review_without_losing_other_claims(self):
        p=passage('The archive probably opened in 1985. The archive provides public catalogue access.')
        cards=[{'id':'E1','passage_id':'S1','quote':'The archive probably opened in 1985.'},
               {'id':'E2','passage_id':'S1','quote':'The archive provides public catalogue access.'}]
        omitted=[]
        claims=bind_evidence_cards({'claims':[{'text':'The archive opened in 1985.','evidence_ids':['E1']},
                                           {'text':'The archive offers public catalogue access.','evidence_ids':['E2']}]},cards,[p],self.config,omitted)
        self.assertEqual(len(claims),1);self.assertEqual(len(omitted),1)
        self.assertIn('public catalogue',claims[0].text)

    def test_hindi_qualifications_are_recognized(self):
        self.assertIn('uncertainty',qualification_flags('यह शायद 1985 में खुला था।'))
        self.assertIn('attribution',qualification_flags('परंपरा के अनुसार इसे व्यास से जोड़ा जाता है।'))
        self.assertIn('negation',qualification_flags('इसका समय निश्चित नहीं है।'))

    def test_short_model_answer_is_checked_by_review_instead_of_extractive_length_rules(self):
        p=passage('The catalogue is public.')
        claims=validate_claims({'claims':[{'text':'Catalogues are public.','evidence':[{'passage_id':'S1','quote':p.text}]}]},[p],self.config)
        self.assertEqual(claims[0].text,'Catalogues are public.')

    def test_no_model_call_is_made_without_usable_evidence(self):
        provider=LocalStub()
        result=Task5Pipeline(self.config,provider=provider).run('What is Mahabharata?',[])
        self.assertEqual(result.generation_mode,'abstained')
        self.assertEqual(provider.prompts,[])

    def test_failed_support_review_cannot_be_presented_as_a_generated_answer(self):
        provider=LocalStub({'claims':[{'text':'The archive offers everyone free entry.','evidence_ids':['E1']}]},{'supported':[False],'conflicts':[]})
        with self.assertRaises(ProviderError):Task5Pipeline(self.config,provider=provider).run('What access exists?',[passage()])

    def test_only_the_supported_statements_survive_a_partial_review(self):
        provider=LocalStub({'claims':[{'text':'Researchers can access public catalogue records.','evidence_ids':['E1']},
                                     {'text':'Everyone gets free entry to the archive.','evidence_ids':['E1']}]},
                           {'supported':[True,False],'conflicts':[]})
        result=Task5Pipeline(self.config,provider=provider).run('What access exists?',[passage()])
        self.assertEqual(len(result.claims),1)
        self.assertNotIn('free entry',result.answer)
        self.assertEqual(result.diagnostics['review']['omitted_proposed_claims'],1)
        self.assertEqual(len(provider.prompts),2)

    def test_bce_ce_translation_cannot_change_the_source_dates(self):
        quote='It was probably compiled between the 3rd century BCE and the 3rd century CE, with oldest parts around 400 BCE.'
        bad='यह शायद 3वीं शताब्दी ईस्वी से 3वीं शताब्दी ईस्वी तक बनाया गया था, जिसमें सबसे पुराना भाग 400 ईस्वी का है।'
        good='यह शायद 3वीं शताब्दी ईसा पूर्व से 3वीं शताब्दी ईस्वी के बीच संकलित हुआ, इसके पुराने भाग लगभग 400 ईसा पूर्व के हैं।'
        self.assertFalse(preserves_numbers_and_eras(bad,[quote]))
        self.assertTrue(preserves_numbers_and_eras(good,[quote]))
        self.assertEqual(date_anchors('३वीं शताब्दी ईसा पूर्व'),[(3,'century','BCE')])

    def test_model_cannot_add_a_number_missing_from_its_card(self):
        self.assertFalse(preserves_numbers_and_eras('The archive opened in 1990.',['The archive opened in 1985.']))

    def test_comparative_date_bound_is_not_replaced_with_unrelated_negation(self):
        source='The oldest parts are not much older than around 400 BCE.'
        self.assertFalse(preserves_numbers_and_eras('सबसे पुराना भाग लगभग 400 ईसा पूर्व से नहीं है।',[source]))
        self.assertFalse(preserves_numbers_and_eras('इसका अधिकांश भाग 3वीं शताब्दी ईसा पूर्व से 3वीं शताब्दी ईस्वी का है और सबसे पुराना भाग 400 ईसा पूर्व से नहीं है।',
                                                  ['Most was compiled in the 3rd century BCE to 3rd century CE, with parts not much older than 400 BCE.']))
        self.assertTrue(preserves_numbers_and_eras('सबसे पुराने भाग लगभग 400 ईसा पूर्व से बहुत पुराने नहीं हैं।',[source]))


if __name__=='__main__':unittest.main()
