"""Evidence context and review regressions from the real Rama/Ravana failure."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from task5 import EvidencePassage,Task5Config,Task5Pipeline
from task5.content import usable_sentences
from task5.evidence import select_evidence
from task5.generation import evidence_cards,bind_evidence_cards
from task5.handoff import adapt_task4_envelope
from task5.providers import ProviderError
from task5.synthesis import draft
from task5.text import normalize

FIXTURE=Path(__file__).parents[1]/'examples/regression_rama_ravana_task4.json'


class Provider:
    name='gemini';model='test-model'
    def __init__(self,*responses):self.responses=iter(responses);self.requests=[]
    def generate(self,prompt,schema,**settings):
        self.requests.append({'prompt':prompt,'schema':schema,**settings})
        return next(self.responses)


def source(text,pid='S1',score=.8,*,html=False):
    return EvidencePassage(pid,text,'unknown',score,.5,'Unknown',language='en',
                           metadata={'source_metadata':{'content_type':'html'}} if html else {})


def review(count=1,coverage='pass',supported=None):
    return {'supported':supported if supported is not None else [True]*count,'relevant':[True]*count,
            'claim_reasons':['The complete cited text supports the statement.']*count,
            'criteria':{'answer_relevance':'pass','evidence_coverage':coverage,'language_fidelity':'pass',
                        'readability':'pass','explanation':'The cited text addresses the question.'},'conflicts':[]}


def nli(claims,*args):
    return {'status':'completed','claims':[{'claim_index':i,'verdict':'entailed','entailment':.9,'contradiction':.01} for i in range(len(claims))],
            'thresholds':{'calibration_status':'uncalibrated_defaults'}}


class EvidenceContextTests(unittest.TestCase):
    def test_real_handoff_keeps_the_named_antecedent_in_the_refusal_quote(self):
        payload=json.loads(FIXTURE.read_text());envelope=payload['task4_output']['queries'][0]
        passages,_=adapt_task4_envelope(envelope);before=copy.deepcopy([p.to_dict() for p in passages])
        config=Task5Config(provider='ollama',min_query_rel=.1,max_passages=4,context_chars=9000)
        selected,_,_=select_evidence(envelope['query'],passages,config)
        cards=evidence_cards(envelope['query'],selected,config)
        cause=next(c for c in cards if 'fought to bring her back after Ravana refused' in c['quote'])
        self.assertIn('Ravana carried Sita away from the forest hermitage at Panchavati.',cause['quote'])
        self.assertEqual(cause['passage_id'],'S3')
        original=next(p for p in passages if p.id=='S3')
        self.assertIn(normalize(cause['quote']),normalize(original.text))
        self.assertEqual([p.to_dict() for p in passages],before)
        self.assertEqual(len(envelope['ranked_evidence']),24)

    def test_context_is_one_contiguous_preceding_sentence(self):
        text='The archive preserves old maps. The catalogue contains twelve entries. It offers public access to these records.'
        rows=list(usable_sentences(source(text),include_antecedent=True))
        self.assertEqual(rows[-1][1],'The catalogue contains twelve entries. It offers public access to these records.')
        self.assertNotIn('preserves old maps',rows[-1][1])
        self.assertTrue(all(normalize(quote) in normalize(text) for _,quote in rows))

    def test_context_does_not_bridge_html_paragraphs_questions_or_headings(self):
        first='The archive preserves old maps.';last='It offers public access to these records.'
        for separator in ['\n',' Who manages the archive? ',' Catalogue details\n']:
            with self.subTest(separator=separator):
                rows=list(usable_sentences(source(first+separator+last,html=True),include_antecedent=True))
                self.assertEqual(rows[-1][1],last)

    def test_context_does_not_reintroduce_page_controls(self):
        text='Sign in to view the catalogue. It offers public access to these records.'
        rows=list(usable_sentences(source(text),include_antecedent=True))
        self.assertEqual(rows,[(text.split('. ',1)[1],text.split('. ',1)[1])])

    def test_context_keeps_source_uncertainty_and_number_guards(self):
        p=source('The cache may be incomplete. It includes 12 public records in its catalogue.')
        config=Task5Config(provider='ollama')
        cards=evidence_cards('What does the cache include?',[p],config)
        contextual=next(c for c in cards if 'It includes' in c['quote'])
        self.assertIn('uncertainty',contextual['qualifications'])
        for text in ['The cache includes 12 public records.','The cache may include 20 public records.']:
            with self.subTest(text=text),self.assertRaises(ProviderError):
                bind_evidence_cards({'claims':[{'text':text,'evidence_ids':[contextual['id']]}]},cards,[p],config)

    def test_local_profile_requires_one_card_per_atomic_claim(self):
        p=source('The archive preserves old maps. The archive offers public catalogue access.')
        config=Task5Config(provider='ollama',verification_profile='standard')
        cards=evidence_cards('What does the archive provide?',[p],config)
        provider=Provider({'claims':[{'text':'The archive preserves maps and offers public catalogue access.',
                                      'evidence_ids':[cards[0]['id'],cards[1]['id']]}]})
        provider.name='ollama'
        with self.assertRaises(ProviderError):draft('What does the archive provide?',cards,[p],'en',config,provider,[])
        schema=provider.requests[0]['schema']['properties']['claims']['items']['properties']
        self.assertEqual(schema['evidence_ids']['maxItems'],1)


class ReviewCoverageTests(unittest.TestCase):
    def test_multi_quote_judge_receives_one_complete_cited_premise_per_claim(self):
        passages=[source('The archive preserves old maps.'),source('The archive offers public catalogue access.','S2')]
        response={'claims':[{'text':'The archive preserves maps and offers public catalogue access.','evidence_ids':['E1','E2']}]}
        provider=Provider(response,review())
        config=Task5Config(provider='gemini',verification_profile='standard',allow_fallback=False)
        with patch('task5.synthesis.check_claims',side_effect=nli):
            result=Task5Pipeline(config,provider=provider).run('What does the archive provide?',passages)
        payload=json.loads(provider.requests[1]['prompt'].split('\n',1)[1])
        item=payload['claims'][0]
        self.assertEqual(item['claim_index'],0)
        self.assertTrue(all(e.quote in item['cited_text'] for e in result.claims[0].evidence))
        self.assertEqual(set(item['citation_passage_ids']),{'S1','S2'})
        self.assertEqual([r['purpose'] for r in provider.requests],['draft','rubric_judge'])

    def run_low_score(self,coverage='pass',supported=None):
        quote='Rama fought Ravana to bring Sita back after Ravana refused to return her.'
        response={'claims':[{'text':quote,'evidence_ids':['E1']}]}
        if supported is not None:response['claims'].append(copy.deepcopy(response['claims'][0]))
        provider=Provider(response,review(len(response['claims']),coverage,supported))
        config=Task5Config(provider='gemini',verification_profile='standard',allow_fallback=False,min_query_rel=.1)
        with patch('task5.synthesis.check_claims',side_effect=nli):
            return Task5Pipeline(config,provider=provider).run('Why did Rama and Ravana fight each other?',
                                                           [source(quote,score=.119)])

    def test_passed_coverage_is_not_overridden_by_low_uncalibrated_scores(self):
        result=self.run_low_score()
        self.assertFalse(result.factors['document_topic_match'])
        self.assertEqual(result.answer_status,'answered')
        self.assertNotIn('Limited evidence',result.answer)
        self.assertEqual(result.reliability.value,'LOW')
        self.assertEqual(result.used_passages[0].query_rel,.119)
        self.assertIn('model judged the cited answer relevant',result.reliability_explanation)

    def test_partial_coverage_stays_limited(self):
        self.assertEqual(self.run_low_score('partial').answer_status,'limited')

    def test_reviewing_a_larger_draft_does_not_establish_remaining_claim_coverage(self):
        result=self.run_low_score(supported=[True,False])
        self.assertEqual(len(result.claims),1)
        self.assertEqual(result.answer_status,'limited')


if __name__=='__main__':unittest.main()
