"""Support-label, evidence-selection and abstention contracts for the pilot."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from task5 import Task5Config,Task5Pipeline,EvidencePassage
from task5.content import usable_sentences
from task5.evidence import select_evidence,detect_conflicts,scope_answer_conflicts
from task5.handoff import adapt_task4_envelope
from task5.intent import requirements
from task5.planning import select_for_question
from task5.generation import evidence_cards,qualification_flags
from task5.providers import ProviderError
from task5.reliability import assess
from task5.schemas import GroundedClaim,ClaimEvidence
from task5.support import assess_support
from task5.text import normalize


def passage(pid='S1',**changes):
    values=dict(id=pid,text='The archive provides public catalogue access.',source_type='unknown',query_rel=.12,
                india_rel=.2,source_signal='Unknown',language='en',url_or_provenance='https://archive.example.org')
    values.update(changes);return EvidencePassage(**values)


def judgment(**changes):
    value={'status':'completed','profile':'standard','reviewed_claims':1,'accepted_claims':1,
        'supported':[True],'relevant':[True],'claim_reasons':['The cited statement directly supports the answer.'],
        'criteria':{'answer_relevance':'pass','evidence_coverage':'pass','language_fidelity':'pass','readability':'pass','explanation':'The question is answered.'},
        'support_assessment':{'label':'High','grounding':'strong','question_coverage':'complete','source_quality':'adequate',
                              'evidence_agreement':'not_assessable','explanation':'The complete answer is directly supported by its quoted reference.'},
        'nli':{'status':'completed','accepted_claim_indices':[0],'claims':[{'claim_index':0,'verdict':'entailed'}],
               'thresholds':{'calibration_status':'uncalibrated_defaults'}},
        'consistency':{'status':'not_run','claims':[]},'conflicts':[]}
    value.update(changes);return value


class Provider:
    name='gemini';model='unit-test'
    def __init__(self,*responses):self.responses=iter(responses);self.requests=[]
    def generate(self,prompt,schema,**options):self.requests.append({'prompt':prompt,'schema':schema,**options});return next(self.responses)


class SupportLabelTests(unittest.TestCase):
    def setUp(self):
        self.p=passage();self.claim=GroundedClaim('The archive provides public catalogue access.',[ClaimEvidence('S1',self.p.text)],'model_reviewed')
    def evaluate(self,review=None,**changes):
        return assess_support([self.claim],[self.p],changes.get('conflicts',[]),review or judgment(),changes.get('quote_coverage',1))

    def test_well_supported_answer_can_be_high_without_fabricating_human_calibration(self):
        label,why,factors=assess([self.claim],[self.p],[self.p],[],judgment())
        self.assertEqual(label.value,'HIGH')
        self.assertIn('model-judged',why)
        self.assertFalse(factors['independence_verified'])
        self.assertEqual(factors['query_relevance'],.12)
        self.assertIsNone(factors['llm_support_assessment']['truth_probability'])
        self.assertFalse(factors['llm_support_assessment']['calibration']['human_review_attested'])

    def test_high_judge_label_cannot_override_missing_citations(self):
        self.assertEqual(self.evaluate(quote_coverage=0)['label'],'LOW')

    def test_high_judge_label_cannot_override_unresolved_conflicts(self):
        self.assertEqual(self.evaluate(conflicts=[{'passage_ids':['S1','S2']}])['label'],'LOW')

    def test_high_judge_label_cannot_override_irrelevance(self):
        review=judgment();review['criteria']['answer_relevance']='fail'
        self.assertEqual(self.evaluate(review)['label'],'LOW')

    def test_high_judge_label_cannot_override_nli_contradiction(self):
        review=judgment();review['nli']['claims'][0]['verdict']='contradicted'
        self.assertEqual(self.evaluate(review)['label'],'LOW')

    def test_weak_ocr_is_a_low_support_guard(self):
        self.p=passage(ocr_conf=.6,extraction_method='OCR')
        self.assertEqual(self.evaluate()['label'],'LOW')

    def test_partial_coverage_or_unknown_source_quality_is_medium(self):
        for field,value in [('question_coverage','partial'),('source_quality','unknown')]:
            review=judgment();review['support_assessment'][field]=value
            with self.subTest(field=field):self.assertEqual(self.evaluate(review)['label'],'MEDIUM')

    def test_a_relevant_but_incomplete_list_is_medium_rather_than_irrelevant(self):
        review=judgment();review['criteria'].update(answer_relevance='partial',evidence_coverage='partial')
        review['support_assessment'].update(label='Medium',question_coverage='partial')
        self.assertEqual(self.evaluate(review)['label'],'MEDIUM')

    def test_a_fitted_nli_requirement_limits_borderline_high_without_rejecting_the_answer(self):
        review=judgment();review['nli']['claims'][0]['verdict']='borderline'
        policy={'high_requires_nli_entailment':True,'calibration':{'status':'model_judged_pilot','human_review_attested':False}}
        with patch('task5.support.support_policy',return_value=policy):self.assertEqual(self.evaluate(review)['label'],'MEDIUM')

    def test_old_results_without_an_llm_assessment_do_not_gain_high_labels(self):
        review=judgment();review.pop('support_assessment')
        self.assertIsNone(self.evaluate(review))

    def test_known_weak_local_self_judge_cannot_assign_high(self):
        review=judgment();review['judge_model']='qwen3:4b-instruct-2507-q4_K_M'
        policy={'high_requires_nli_entailment':False,'high_blocked_judge_models':[review['judge_model']],
                'calibration':{'status':'model_judged_pilot','human_review_attested':False}}
        with patch('task5.support.support_policy',return_value=policy):self.assertEqual(self.evaluate(review)['label'],'MEDIUM')

    def test_cross_provider_review_keeps_qwen_generation_and_uses_gemini_judgment(self):
        primary=Provider({'claims':[{'text':self.p.text,'evidence_ids':['E1']}]});primary.name='ollama';primary.model='qwen3:4b-instruct-2507-q4_K_M'
        judge=Provider(judgment());judge.name='gemini';judge.model='gemini-3.5-flash-lite'
        config=Task5Config(provider='ollama',judge_provider='gemini',verification_profile='standard',min_query_rel=.1,allow_fallback=False)
        with patch('task5.synthesis.StructuredProvider',return_value=judge),patch('task5.synthesis.check_claims',return_value=judgment()['nli']):
            result=Task5Pipeline(config,provider=primary).run('What catalogue access does the archive offer?',[self.p])
        self.assertEqual(result.generation_mode,'ollama')
        self.assertEqual(result.reliability.value,'HIGH')
        self.assertEqual(result.diagnostics['review']['judge_provider'],'gemini')
        self.assertFalse(result.diagnostics['review']['judge_is_drafting_model'])
        self.assertEqual([r['purpose'] for r in primary.requests],['draft'])
        self.assertEqual([r['purpose'] for r in judge.requests],['rubric_judge'])


class QuestionSelectionTests(unittest.TestCase):
    def test_planning_preserves_ids_scores_and_source_caps(self):
        sources=[passage('S'+str(i)) for i in range(1,5)]
        sources[3].url_or_provenance='https://other.example.net'
        original=copy.deepcopy([p.to_dict() for p in sources])
        provider=Provider({'passage_ids':['S1','S2','S3','S4'],'answerability':'sufficient','reason':'These passages cover the requested access.'})
        result,report=select_for_question('What access exists?',sources,sources[:1],'en',Task5Config(max_passages=4),provider)
        self.assertEqual([p.id for p in result],['S1','S2','S4'])
        self.assertEqual([p.to_dict() for p in sources],original)
        self.assertTrue(report['original_task4_ranks_and_scores_preserved'])

    def test_selection_rejects_unknown_duplicate_and_non_string_ids(self):
        for ids in [['unknown'],['S1','S1'],[{}]]:
            provider=Provider({'passage_ids':ids,'answerability':'sufficient','reason':'A reason.'})
            with self.subTest(ids=ids),self.assertRaises(ProviderError):
                select_for_question('Question',[passage()],[passage()],'en',Task5Config(),provider)

    def test_standard_path_has_one_selection_draft_and_judge_call(self):
        provider=Provider({'passage_ids':['S1'],'answerability':'sufficient','reason':'The quoted access policy answers the question.'},
                          {'claims':[{'text':passage().text,'evidence_ids':['E1']}]},judgment())
        config=Task5Config(provider='gemini',verification_profile='standard',query_focused_selection=True,min_query_rel=.1,allow_fallback=False)
        with patch('task5.synthesis.check_claims',return_value=judgment()['nli']):
            result=Task5Pipeline(config,provider=provider).run('What access does the archive provide?',[passage()])
        self.assertEqual([r['purpose'] for r in provider.requests],['evidence_selection','draft','rubric_judge'])
        self.assertEqual(result.reliability.value,'HIGH')

    def test_valid_evidence_abstention_is_a_result_not_an_operational_failure(self):
        provider=Provider({'passage_ids':[],'answerability':'insufficient','reason':'The passage does not establish the requested opening date.'})
        config=Task5Config(provider='gemini',query_focused_selection=True,min_query_rel=.1,allow_fallback=False)
        result=Task5Pipeline(config,provider=provider).run('When did the archive open?',[passage()])
        self.assertEqual(result.answer_status,'insufficient_evidence')
        self.assertEqual(result.reliability.value,'LOW');self.assertEqual(len(provider.requests),1)

    def test_empty_model_draft_can_be_reviewed_as_insufficient_evidence(self):
        review=judgment(reviewed_claims=0,accepted_claims=0,supported=[],relevant=[],claim_reasons=[])
        review['criteria']['evidence_coverage']='fail';review['support_assessment'].update(label='Low',grounding='weak',question_coverage='not_answered')
        provider=Provider({'claims':[]},review)
        config=Task5Config(provider='gemini',verification_profile='standard',min_query_rel=.1,allow_fallback=False)
        result=Task5Pipeline(config,provider=provider).run('When did the archive open?',[passage()])
        self.assertEqual(result.answer_status,'insufficient_evidence')
        self.assertEqual(result.claims,[])

    def test_low_lexical_scores_do_not_discard_real_cause_before_the_llm_sees_it(self):
        fixture=Path(__file__).parents[1]/'examples/regression_qwen_why_task4.json'
        envelope=json.loads(fixture.read_text())['task4_output']['queries'][0]
        passages,_=adapt_task4_envelope(envelope)
        for p in passages:p.metadata['task4_retrieval_mode']='lexical'
        selected,pool,_=select_evidence(envelope['query'],passages,Task5Config(provider='ollama',min_query_rel=.1,max_passages=4))
        self.assertIn('S6',{p.id for p in pool})
        self.assertEqual(next(p.query_rel for p in pool if p.id=='S6'),.067)
        self.assertTrue(any('abducted' in p.text for p in selected))

    def test_legacy_word_wrapped_html_reflows_without_changing_original_quotes(self):
        text='The\nGreen\nRevolution\nin\nIndia\nused\nhigh-yielding\nseeds\nand\nimproved\nirrigation\nto\nincrease\ncrop\nproduction.'
        p=passage(text=text,metadata={'source_metadata':{'content_type':'html'}})
        rows=list(usable_sentences(p))
        self.assertEqual(len(rows),1)
        self.assertEqual(normalize(rows[0][1]),normalize(text))

    def test_requested_counts_and_reason_requirements_are_explicit(self):
        self.assertEqual(requirements('Give three facts about the Ramayana')['requested_count'],3)
        self.assertEqual(requirements('रामायण के बारे में तीन तथ्य बताओ')['requested_count'],3)
        self.assertEqual(requirements('राम और रावण के बीच युद्ध क्यों हुआ?')['kind'],'reason')

    def test_a_specific_hindi_name_question_uses_the_validated_source_focus(self):
        from task5.content import overview_query
        query='हनुमान की माता का नाम क्या बताया गया है?'
        self.assertFalse(overview_query(query))
        quote='Hanuman is believed to have been fondly called Sundara by his mother Anjani.'
        p=passage(text='Sundara Kanda is the fifth book of the Ramayana. It describes the adventures of Hanuman. The work emphasizes his devotion to Rama. '+quote,
                  metadata={'upstream_passage_id':'original','task4_retrieval_mode':'lexical','task5_focus_quotes':[quote]})
        cards=evidence_cards(query,[p],Task5Config(provider='gemini'))
        self.assertIn('mother Anjani',cards[0]['quote'])
        self.assertEqual(cards[0]['evidence_role'],'question_focused')

    def test_exact_focused_table_facts_survive_but_controls_and_absent_quotes_do_not(self):
        quote='M. S. Swaminathan, the Father of the Green Revolution in India'
        p=passage(text='Contributors\n'+quote+'\nThank you for your feedback!',
                  metadata={'task5_focus_quotes':[quote,'Thank you for your feedback!','An absent invented fact about farming.']})
        cards=evidence_cards('Who is the Father of the Green Revolution in India?',[p],Task5Config(provider='gemini'))
        self.assertEqual(len(cards),1);self.assertEqual(cards[0]['quote'],quote)

    def test_natural_hindi_reported_attribution_is_preserved(self):
        for text in ['हनुमान की माता का नाम अंजनी बताया गया है।','ऐसा कहा जाता है।','उन्हें लेखक माने जाते हैं।']:
            with self.subTest(text=text):self.assertIn('attribution',qualification_flags(text))
        self.assertNotIn('attribution',qualification_flags('हनुमान की माता अंजनी हैं।'))
        self.assertNotIn('uncertainty',qualification_flags('यह बताया गया है।'))

    def test_a_contained_quote_with_an_extra_record_number_is_not_a_contradiction(self):
        a=passage(text='Record 5: The River Archive opened in 1985.')
        b=passage('S2',text='The River Archive opened in 1985.')
        self.assertEqual(detect_conflicts([a,b]),[])

    def test_containment_does_not_hide_opposing_negation(self):
        a=passage(text='The archive offers free catalogue access.')
        b=passage('S2',text='It is not true: The archive offers free catalogue access.')
        self.assertTrue(detect_conflicts([a,b]))

    def test_unrelated_numeric_disputes_are_retained_as_background_not_answer_conflicts(self):
        conflicts=[{'quotes':['The archive opened in 1985.','The archive opened in 1990.']}]
        author=GroundedClaim('Mira wrote the catalogue.',[])
        relevant,background=scope_answer_conflicts(conflicts,[author],'Who wrote the catalogue?')
        self.assertEqual(relevant,[]);self.assertEqual(background,conflicts)
        dated=GroundedClaim('The archive opened in 1985.',[])
        self.assertEqual(scope_answer_conflicts(conflicts,[dated],'When did the archive open?')[0],conflicts)


if __name__=='__main__':unittest.main()
