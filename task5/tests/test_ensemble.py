"""Meaningful ensemble contracts; real model smoke tests are recorded separately."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from task5 import Task5Config,Task5Pipeline,EvidencePassage
from task5.providers import ProviderError
from task5.schemas import GroundedClaim,ClaimEvidence
from task5.synthesis import ordered_cards
from task5.verification import retrieval_diagnostics,consistency_signal
from task5.experiments import plan
from task5.human_review import export_tasks,calibrate,reviewed_rows,digest
from task5.nli import MODEL,REVISION,thresholds


def source(text='The archive provides public catalogue access.',pid='S1',score=.9):
    return EvidencePassage(pid,text,'institutional',score,.6,'High',language='en',url_or_provenance='https://archive.example.org')


def judgment(count=1,**changes):
    value={'supported':[True]*count,'relevant':[True]*count,'claim_reasons':['The cited statement supports this claim.']*count,
           'criteria':{'answer_relevance':'pass','evidence_coverage':'partial','language_fidelity':'pass','readability':'pass','explanation':'Only the available evidence is covered.'},'conflicts':[]}
    value.update(changes);return value


def nli_result(verdict='entailed'):
    return {'status':'completed','claims':[{'claim_index':0,'verdict':verdict,'entailment':.9 if verdict=='entailed' else .1,'contradiction':.9 if verdict=='contradicted' else .1}],
            'thresholds':{'calibration_status':'uncalibrated_defaults'}}


class Provider:
    name='gemini';model='test-gemini'
    def __init__(self,*responses):self.responses=iter(responses);self.requests=[]
    def generate(self,prompt,schema,**settings):
        self.requests.append({'prompt':prompt,'schema':schema,**settings})
        return next(self.responses)


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.config=Task5Config(provider='gemini',model='test-gemini',verification_profile='standard',allow_fallback=False)
        self.draft={'claims':[{'text':'Researchers can access public catalogue records.','evidence_ids':['E1']}]}

    def run_model(self,provider,verdict='entailed',language='en'):
        with patch('task5.synthesis.check_claims',return_value=nli_result(verdict)):
            return Task5Pipeline(self.config,provider=provider).run('What access does the archive provide?',[source()],target_language=language)

    def test_gemini_uses_server_bound_quotes_and_a_complete_rubric(self):
        provider=Provider(self.draft,judgment());result=self.run_model(provider)
        self.assertEqual(result.claims[0].evidence[0].quote,source().text)
        self.assertEqual(result.diagnostics['review']['nli']['status'],'completed')
        self.assertEqual(result.factors['judge_criteria']['answer_relevance'],'pass')
        self.assertEqual([r['purpose'] for r in provider.requests],['draft','rubric_judge'])
        self.assertEqual(result.answer_status,'limited')
        self.assertIn('uncalibrated',result.reliability_explanation)

    def test_nli_contradiction_cannot_be_overridden_by_a_positive_self_judge(self):
        with self.assertRaisesRegex(ProviderError,'No answer statement'):
            self.run_model(Provider(self.draft,judgment()),'contradicted')

    def test_borderline_nli_can_reach_the_judge_but_is_disclosed(self):
        result=self.run_model(Provider(self.draft,judgment()),'borderline')
        self.assertIn('borderline',result.reliability_explanation)
        self.assertEqual(len(result.claims),1)

    def test_supported_but_irrelevant_claim_is_rejected(self):
        with self.assertRaises(ProviderError):self.run_model(Provider(self.draft,judgment(relevant=[False])))

    def test_standard_profile_rejects_an_incomplete_judging_rubric(self):
        with self.assertRaisesRegex(ProviderError,'complete rubric'):
            self.run_model(Provider(self.draft,judgment(criteria={})))

    def test_wrong_language_cannot_be_approved_by_the_judge(self):
        hindi={'claims':[{'text':'अभिलेखागार सार्वजनिक सूची तक पहुँच देता है।','evidence_ids':['E1']}]}
        with self.assertRaisesRegex(ProviderError,'language'):
            self.run_model(Provider(hindi,judgment()))

    def test_thorough_profile_resamples_at_nonzero_temperature(self):
        self.config=Task5Config(provider='gemini',verification_profile='thorough',allow_fallback=False)
        provider=Provider(self.draft,self.draft,judgment())
        with patch('task5.nli.readiness',return_value={'ready':False}):result=self.run_model(provider)
        self.assertEqual([r['purpose'] for r in provider.requests],['draft','consistency_draft','rubric_judge'])
        self.assertGreater(provider.requests[1]['temperature'],0)
        self.assertEqual(result.diagnostics['review']['consistency']['additional_drafts'],1)

    def test_translate_first_draft_sees_translations_and_cites_originals(self):
        self.config=Task5Config(provider='gemini',verification_profile='standard',cross_lingual_strategy='translate_then_summarize',allow_fallback=False)
        text='अभिलेखागार सार्वजनिक सूची तक पहुँच देता है।'
        provider=Provider({'translations':[{'evidence_id':'E1','text':text}]},
                          {'claims':[{'text':text,'evidence_ids':['E1']}]},judgment())
        result=self.run_model(provider,language='hi')
        self.assertEqual(provider.requests[0]['purpose'],'translation')
        self.assertNotIn(source().text,provider.requests[1]['prompt'])
        self.assertIn(text,provider.requests[1]['prompt'])
        self.assertEqual(result.claims[0].evidence[0].quote,source().text)

    def test_reversing_passage_order_keeps_ids_and_quotes_intact(self):
        cards=[{'id':'E1','passage_id':'S1','quote':'first'},{'id':'E2','passage_id':'S2','quote':'second'},{'id':'E3','passage_id':'S1','quote':'third'}]
        result=ordered_cards('query',cards,[source(),source(pid='S2')],'reverse')
        self.assertEqual([row['id'] for row in result],['E2','E1','E3'])
        self.assertEqual({row['id']:row['quote'] for row in result},{row['id']:row['quote'] for row in cards})

    def test_flat_high_relevance_is_not_automatically_called_irrelevant(self):
        high=[source(pid='S'+str(i),score=.9) for i in range(3)]
        report=retrieval_diagnostics(high,high[:2])
        self.assertTrue(report['flat_scores']);self.assertFalse(report['low_relevance'])
        low=[source(pid='S'+str(i),score=.1) for i in range(3)]
        self.assertTrue(retrieval_diagnostics(low,[])['low_relevance'])

    def test_a_claim_missing_from_the_alternative_is_flagged(self):
        claim=GroundedClaim('The archive provides catalogue access.',[ClaimEvidence('S1',source().text)])
        other=GroundedClaim('The archive closes on Sundays.',[ClaimEvidence('S1',source().text)])
        with patch('task5.nli.readiness',return_value={'ready':False}):
            value=consistency_signal([claim],[[other]],self.config)
        self.assertTrue(value['claims'][0]['unstable'])
        self.assertEqual(value['matching_method'],'exact_text_only')
        self.assertIn('not',value['interpretation'])

    def test_experiment_plan_preserves_upstream_ranking_and_bounds_calls(self):
        p=Path(__file__).parents[1]/'examples/contract_example.json';payload=json.loads(p.read_text())
        before=copy.deepcopy(payload)
        rows,jobs=plan(payload,[2,4],['relevance','reverse'],['direct'],max_runs=4)
        self.assertEqual(len(jobs),4);self.assertEqual(payload,before)
        with self.assertRaises(ValueError):plan(payload,[2,4],['relevance','reverse'],['direct'],max_runs=3)


class HumanCalibrationTests(unittest.TestCase):
    def test_blind_export_includes_rejected_claims_without_showing_machine_verdicts(self):
        p=source().to_dict();evidence=[{'passage_id':'S1','quote':p['text']}]
        decisions=[{'claim_index':0,'text':'The catalogue is public.','evidence':evidence,'supported':True,'accepted':True},
                   {'claim_index':1,'text':'The catalogue costs money.','evidence':evidence,'supported':False,'accepted':False}]
        result={'target_language':'en','claims':[decisions[0]],'used_passages':[p],'selected_passages':[p],
                'diagnostics':{'review':{'claim_decisions':decisions}}}
        tasks,signals=export_tasks([{'task5_output':{'queries':[{'query':'What access exists?','result':result}]}}])
        self.assertEqual(len(tasks),2)
        self.assertTrue(all(row['review_status']=='pending' and row['human_review_attested'] is False for row in tasks))
        self.assertTrue(all('accepted_by_pipeline' not in row and 'judge_supported' not in row for row in tasks))
        self.assertFalse(signals['signals'][tasks[1]['task_id']]['accepted_by_pipeline'])

    def pending_data(self,count=24):
        tasks=[];signals={}
        groups=sorted(['group-'+str(i) for i in range(count)],key=digest)
        for index,group in enumerate(groups):
            supported=index%4<2;task_id='task-'+str(index)
            tasks.append({'schema_version':'task5.human_review.v1','task_id':task_id,'query_group':group,'query':group,
                          'language':'en','claim':'A claim.','references':[],'review_status':'pending','reviewer_id':None,
                          'human_review_attested':False,'faithfulness':None,'relevant':None,'language_fidelity':None})
            signals[task_id]={'language':'en','query_group':group,'nli_model':MODEL,'nli_revision':REVISION,
                             'nli_entailment':.95 if supported else .1,'nli_contradiction':.1 if supported else .95,
                             'judge_supported':supported,'consistency':1.0 if supported else 0.0}
        return tasks,signals

    def reviews(self,tasks,signals):
        # Explicitly synthetic validator fixtures, never persisted as real annotations.
        rows=copy.deepcopy(tasks)
        for row in rows:row.update(review_status='complete',reviewer_id='unit-test-fixture',human_review_attested=True,
                                  faithfulness='supported' if signals[row['task_id']]['judge_supported'] else 'contradicted',relevant=True,language_fidelity='pass')
        return rows

    def test_pending_reviews_do_not_create_calibrated_thresholds(self):
        tasks,signals=self.pending_data();report=calibrate(tasks,tasks,signals)
        self.assertEqual(report['reviewed_items'],0)
        self.assertEqual(report['languages']['en']['status'],'insufficient_reviewed_data')

    def test_claim_edits_or_unattested_reviews_cannot_enter_calibration(self):
        tasks,signals=self.pending_data();reviews=self.reviews(tasks,signals)
        reviews[0]['claim']='Changed claim'
        with self.assertRaises(ValueError):reviewed_rows(tasks,reviews)
        reviews=self.reviews(tasks,signals);reviews[0]['human_review_attested']=False
        with self.assertRaises(ValueError):reviewed_rows(tasks,reviews)

    def test_calibration_uses_disjoint_questions_and_reports_all_three_signals(self):
        tasks,signals=self.pending_data();report=calibrate(tasks,self.reviews(tasks,signals),signals)
        profile=report['languages']['en'];self.assertEqual(profile['status'],'reviewed_holdout_evaluated')
        self.assertFalse(set(profile['train_question_groups'])&set(profile['heldout_question_groups']))
        self.assertEqual(set(profile['heldout_metrics']),{'nli','sampling_consistency','judge','all_three'})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic-calibration.json';path.write_text(json.dumps(report))
            config=Task5Config(calibration_path=str(path))
            self.assertEqual(thresholds(config,'en')['calibration_status'],'reviewed_holdout_evaluated')
            self.assertNotEqual(thresholds(config,'hi')['calibration_status'],'reviewed_holdout_evaluated')

    def test_unsupported_is_not_automatically_labelled_contradiction(self):
        tasks,signals=self.pending_data();reviews=self.reviews(tasks,signals)
        for row in reviews:
            if row['faithfulness']=='contradicted':row['faithfulness']='unsupported'
        report=calibrate(tasks,reviews,signals)
        self.assertEqual(report['languages']['en']['status'],'reviewed_holdout_evaluated')
        self.assertEqual(report['languages']['en']['contradiction_threshold_status'],'default_no_contradiction_examples')


if __name__=='__main__':unittest.main()
