"""User-language precedence, bounded point requests and output-language guards."""
import unittest
from unittest.mock import patch
from task5 import Task5Config, Task5Pipeline, EvidencePassage
from task5.intent import requested_count
from task5.language import explicit_answer_language, resolve_answer_language
from task5.schemas import ClaimEvidence, GroundedClaim
from task5.synthesis import check_language
from task5.providers import ProviderError
from task5.text import detect_language


class LanguageRequestTests(unittest.TestCase):
    def test_explicit_language_overrides_stale_selector(self):
        queries={
            'tell me 5 points on ramayana in hindi':'hi',
            'Please answer in Hindi':'hi',
            'रामायण की पाँच बातें हिंदी में बताइए':'hi',
            'हिन्दी में समझाइए':'hi',
            'hindi mein batao':'hi',
            'English mein batao':'en',
            'रामायण के बारे में बताइए, answer in English':'en',
            'Translate Hindi into English':'en',
            'In Hindi, explain the Ramayana':'hi',
            'Output language: Hindi':'hi',
        }
        for query,expected in queries.items():
            with self.subTest(query=query):
                self.assertEqual(explicit_answer_language(query),expected)
                self.assertEqual(resolve_answer_language(query,'en' if expected=='hi' else 'hi')['resolved_language'],expected)

    def test_topic_names_quotes_and_negation_are_not_positive_instructions(self):
        for query in ['Tell me about Hindi literature','Describe the English language',
                      'What does "in Hindi" mean?',"Explain the phrase 'in Hindi'",'Do not answer in Hindi']:
            with self.subTest(query=query):self.assertIsNone(explicit_answer_language(query))
        self.assertEqual(explicit_answer_language('Explain in Hindi, not English'),'hi')
        self.assertEqual(explicit_answer_language('हिंदी में नहीं, अंग्रेज़ी में जवाब दीजिए'),'en')

    def test_last_explicit_instruction_wins(self):
        self.assertEqual(explicit_answer_language('Answer in Hindi. Actually answer in English.'),'en')

    def test_selector_and_query_script_are_fallbacks(self):
        self.assertEqual(resolve_answer_language('रामायण क्या है?','en')['resolved_language'],'en')
        self.assertEqual(resolve_answer_language('रामायण क्या है?')['resolved_language'],'hi')
        self.assertEqual(resolve_answer_language('What is the Ramayana?')['resolved_language'],'en')
        # Source-language detection must not start interpreting passage wording.
        self.assertEqual(detect_language('The article says answer in Hindi.'),'en')

    def test_hindi_guard_rejects_english_with_only_a_hindi_name(self):
        for text in ['The Ramayana is an epic.','The Ramayana (रामायण) is an ancient Sanskrit epic.']:
            with self.subTest(text=text),self.assertRaises(ProviderError):
                check_language([GroundedClaim(text,[])],'hi')
        check_language([GroundedClaim('रामायण संस्कृत का एक प्राचीन महाकाव्य है।',[])],'hi')

    def test_count_requires_a_requested_list_not_a_date_or_ordinal(self):
        for query in ['tell me 5 points on ramayana in hindi','Give five key facts about Rama',
                      'रामायण के बारे में पाँच बातें बताओ','रामायण के ५ मुख्य बिंदु बताइए']:
            with self.subTest(query=query):self.assertEqual(requested_count(query),5)
        for query in ['What happened in 2 BCE?','Who won 3 elections?','Tell me about Task 4']:
            with self.subTest(query=query):self.assertIsNone(requested_count(query))

    def run_points(self,accepted=5,requested=5):
        text='The museum preserves regional records. The museum displays sculptures. The museum hosts workshops.'
        passage=EvidencePassage('S1',text,'institutional',.9,.8,'High',language='en',url_or_provenance='https://museum.example.org')
        claims=[GroundedClaim('संग्रहालय में क्षेत्रीय अभिलेख सुरक्षित रखे जाते हैं।',
                              [ClaimEvidence('S1','The museum preserves regional records.')],'model_reviewed') for _ in range(accepted)]
        review={'status':'completed','profile':'basic',
                'criteria':{'answer_relevance':'pass','evidence_coverage':'pass'},
                'support_assessment':{'label':'High','grounding':'strong','question_coverage':'complete',
                                      'source_quality':'adequate','evidence_agreement':'not_assessable','explanation':'Test judge proposes complete support.'}}
        config=Task5Config(provider='gemini',max_claims=3,allow_fallback=False)
        with patch('task5.pipeline.synthesize',return_value=(claims,'gemini',[],[],review)) as mock:
            result=Task5Pipeline(config,provider=object()).run(f'Tell me {requested} points about the museum in Hindi',[passage],target_language='en')
        return result,mock.call_args.args

    def test_five_point_budget_and_hindi_reach_synthesis(self):
        result,args=self.run_points()
        self.assertEqual(args[3],'hi');self.assertEqual(args[4].max_claims,5)
        self.assertEqual(result.target_language,'hi')
        self.assertEqual(len(result.claims),5)
        self.assertIn('\n5. ',result.answer)
        self.assertEqual(result.diagnostics['language_resolution']['basis'],'explicit_query_instruction')

    def test_partial_point_count_cannot_receive_high_even_if_judge_says_high(self):
        result,_=self.run_points(accepted=2)
        self.assertEqual(result.answer_status,'limited')
        self.assertEqual(result.reliability.value,'MEDIUM')
        self.assertEqual(result.factors['llm_support_assessment']['label'],'MEDIUM')
        self.assertIn('सीमित साक्ष्य',result.answer)

    def test_large_list_request_keeps_bounded_budget_and_marks_shortfall(self):
        result,args=self.run_points(accepted=2,requested=20)
        self.assertEqual(args[4].max_claims,8)
        self.assertEqual(result.answer_status,'limited')


if __name__=='__main__':unittest.main()
