"""Regression tests for real low-score Sundara Kanda evidence and coverage reporting."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from task5 import Task5Config,Task5Pipeline,EvidencePassage
from task5.handoff import adapt_task4_envelope,summarize_task4_payload
from task5.evidence import select_evidence
from task5.generation import evidence_cards,bind_evidence_cards
from task5.providers import ProviderError
from task5.text import query_overlap,sentences,query_terms
from task5.content import sentence_match

FIXTURE=Path(__file__).parents[1]/'examples/regression_sundarakanda_task4.json'


class TopicCoverageTests(unittest.TestCase):
    def setUp(self):
        self.payload=json.loads(FIXTURE.read_text());self.envelope=self.payload['task4_output']['queries'][0]
        self.query=self.envelope['query'];self.passages,_=adapt_task4_envelope(self.envelope)
        self.config=Task5Config(provider='extractive',min_query_rel=.1,max_passages=4)

    def test_compound_name_variants_match_both_directions_without_changing_constraints(self):
        for query,text in [('sundarakanda','Sundara Kāṇḍa'),('sundara kanda','Sundarakanda'),('sundarakanda','Sundara-kanda')]:
            self.assertEqual(query_overlap(query,text),1)
        self.assertEqual(query_overlap('not able','notable'),0)
        self.assertIn('not',query_terms('What is not permitted?'))
        self.assertIn('1950',query_terms('What happened in 1950?'))

    def test_sanskrit_annotation_and_initials_do_not_break_the_definition(self):
        text="Sundara Kanda (Sanskrit: सुन्दरकाण्ड, lit. 'beautiful chapter') is the fifth book of the Ramayana. M. S. Ramarao translated it."
        rows=sentences(text)
        self.assertEqual(len(rows),2);self.assertIn('fifth book',rows[0]);self.assertIn('M. S. Ramarao',rows[1])

    def test_real_introduction_and_plot_survive_weak_lexical_scores(self):
        original=copy.deepcopy([p.to_dict() for p in self.passages])
        selected,_,decisions=select_evidence(self.query,self.passages,self.config)
        ids={p.id for p in selected}
        self.assertTrue({'S5','S6'}<=ids);self.assertNotIn('S2',ids)
        self.assertNotIn('S4',ids);self.assertNotIn('S8',ids);self.assertNotIn('S9',ids)
        by_id={p.id:p for p in selected}
        self.assertEqual(by_id['S5'].query_rel,.027);self.assertEqual(by_id['S6'].query_rel,0)
        self.assertEqual([p.to_dict() for p in self.passages],original)
        self.assertIn('low lexical-score',next(d['reason'] for d in decisions if d['id']=='S6'))

    def test_generation_cards_include_plot_and_preserve_exact_original_quotes(self):
        selected,_,_=select_evidence(self.query,self.passages,self.config)
        cards=evidence_cards(self.query,selected,self.config)
        combined=' '.join(card['statement'] for card in cards)
        self.assertIn('fifth book',combined);self.assertIn('Hanuman',combined);self.assertIn('Sita',combined);self.assertIn('Lanka',combined)
        self.assertTrue(any(c['passage_id']=='S5' and 'leap across the ocean' in c['statement'] for c in cards))
        self.assertNotIn('may also refer',combined);self.assertNotIn('Telugu songs',combined)
        source={p.id:p.text for p in selected}
        self.assertTrue(all(card['quote'] in source[card['passage_id']] for card in cards))
        self.assertEqual(sum(card['evidence_role']=='definition' for card in cards),1)

    def test_same_document_continuation_can_support_a_sentence_without_repeating_the_title(self):
        selected,_,_=select_evidence(self.query,self.passages,self.config)
        plot=next(p for p in selected if p.id=='S6')
        self.assertTrue(sentence_match(self.query,"Hanuman discloses himself to Sita as Rama's messenger.",plot))
        self.assertFalse(sentence_match(self.query,'The giant planet orbits a distant star.',plot))

    def test_title_alone_does_not_rescue_unrelated_text_or_flat_api_scores(self):
        irrelevant=EvidencePassage('S1','The giant planet orbits a distant star.', 'unknown',0,0,'Unknown',language='en',title='Sundara Kanda',metadata={'upstream_passage_id':'original'})
        selected,_,_=select_evidence(self.query,[irrelevant],self.config);self.assertEqual(selected,[])
        flat=EvidencePassage('S1','Sundara Kanda is a book of the Ramayana.', 'unknown',0,0,'Unknown',language='en',title='Sundara Kanda')
        self.assertEqual(select_evidence(self.query,[flat],self.config)[0],[])

    def test_explicit_semantic_handoffs_keep_their_numeric_relevance_gate(self):
        for passage in self.passages:passage.metadata['task4_retrieval_mode']='semantic'
        selected,_,_=select_evidence(self.query,self.passages,self.config)
        self.assertFalse({'S5','S6'}&{p.id for p in selected})

    def test_same_domain_does_not_transfer_context_between_unrelated_documents(self):
        unrelated=EvidencePassage('S100','The giant planet orbits a distant star.', 'unknown',0,0,'Unknown',language='en',url_or_provenance='https://en.wikipedia.org/wiki/Planet',title='Planet',metadata={'upstream_passage_id':'planet','upstream_document_id':'unrelated'})
        selected,_,_=select_evidence(self.query,self.passages+[unrelated],self.config)
        self.assertNotIn('S100',{p.id for p in selected})

    def test_low_raw_rank_scores_do_not_force_a_model_reviewed_answer_to_be_limited(self):
        from task5.schemas import GroundedClaim,ClaimEvidence
        claim=GroundedClaim('Sundara Kanda follows Hanuman’s adventures.',[ClaimEvidence('S5',"The work depicts the adventures of Hanuman and his selflessness, strength, and devotion to Rama are emphasised in the text.")],'model_reviewed')
        review={'status':'completed','criteria':{'evidence_coverage':'pass'}}
        with patch('task5.pipeline.synthesize',return_value=([claim],'gemini',[],[],review)):
            result=Task5Pipeline(self.config).run(self.query,self.passages)
        self.assertEqual(result.answer_status,'answered')
        self.assertNotIn('Limited evidence',result.answer)
        self.assertEqual(result.reliability.value,'LOW')
        self.assertTrue(result.factors['answer_coverage']['topic_matched'])

    def test_a_compound_claim_binds_all_supporting_quotes_without_duplicate_visible_ids(self):
        selected,_,_=select_evidence(self.query,self.passages,self.config)
        cards=evidence_cards(self.query,selected,self.config)
        definition=next(c for c in cards if c['passage_id']=='S5' and 'fifth book' in c['quote'])
        plot=next(c for c in cards if c['passage_id']=='S5' and 'leap across the ocean' in c['quote'])
        text='Sundara Kanda is the fifth book of the Ramayana and describes Hanuman’s leap to Lanka.'
        claims=bind_evidence_cards({'claims':[{'text':text,'evidence_ids':[definition['id'],plot['id']]}]},cards,selected,self.config,allow_multiple=True)
        self.assertEqual(len(claims[0].evidence),2)
        with patch('task5.pipeline.synthesize',return_value=(claims,'gemini',[],[],{'status':'completed','criteria':{'evidence_coverage':'pass'}})):
            result=Task5Pipeline(self.config).run(self.query,self.passages)
        self.assertEqual(result.answer.count('[S5]'),1)
        self.assertEqual(len(result.supporting_sources[0]['quotes']),2)

    def test_repeated_evidence_card_ids_are_rejected_in_multi_card_mode(self):
        selected,_,_=select_evidence(self.query,self.passages,self.config);cards=evidence_cards(self.query,selected,self.config)
        with self.assertRaises(ProviderError):
            bind_evidence_cards({'claims':[{'text':'Sundara Kanda is a book of the Ramayana.','evidence_ids':[cards[0]['id'],cards[0]['id']]}]},cards,selected,self.config,allow_multiple=True)


if __name__=='__main__':unittest.main()
