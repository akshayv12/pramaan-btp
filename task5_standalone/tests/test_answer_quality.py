"""Regression checks for the real Mahabharata failure and neighboring failure modes."""
from pathlib import Path
import json
import unittest

from task5 import EvidencePassage, Task5Config, Task5Pipeline
from task5.handoff import summarize_task4_payload
from task5.text import normalize, query_overlap, query_terms
from task5.content import concise_statement
from task5.reliability import assess
from task5.schemas import GroundedClaim, ClaimEvidence


def source(text, **kwargs):
    return EvidencePassage('S1',text,source_type='unknown',india_rel=0,query_rel=kwargs.pop('query_rel',.8),
                           source_signal='Unknown',language='en',**kwargs)


class AnswerQualityTests(unittest.TestCase):
    def setUp(self):
        self.pipeline=Task5Pipeline(Task5Config(provider='extractive',min_query_rel=.10))

    def test_real_failed_handoff_abstains_instead_of_repeating_quiz_footer(self):
        fixture=Path(__file__).parents[1]/'examples/regression_mahabharata_task4.json'
        output=summarize_task4_payload(json.loads(fixture.read_text()),config=self.pipeline.config)
        row=output['task5_output']['queries'][0]
        self.assertEqual(row['answer_status'],'insufficient_evidence')
        self.assertEqual(row['result']['claims'],[])
        self.assertEqual(len(row['citation_to_task4_passage_id']),8)
        self.assertNotIn('Thank you',row['result']['answer'])
        self.assertNotIn('Bible',row['result']['answer'])

    def test_ui_questions_and_unfinished_lines_are_not_claims_even_with_high_input_score(self):
        for text in ('Thank you for your feedback!', 'Who taught Kunti the mantras that helped her beget children by',
                     'The Ultimate Bible Quiz!', 'Please provide name and email to proceed'):
            with self.subTest(text=text):
                result=self.pipeline.run('what do you know mahabharatha ?',[source(text)])
                self.assertEqual(result.claims,[])

    def test_weak_passage_score_cannot_authorize_an_unrelated_sentence(self):
        result=self.pipeline.run('What is Mahabharata?',[source('The giant planet orbits a distant star every twelve years.',query_rel=.324)])
        self.assertEqual(result.answer_status,'insufficient_evidence')

    def test_spelling_and_latin_diacritics_match_the_actual_subject(self):
        text='The Mahābhārata is one of the Sanskrit epics of ancient India.'
        result=self.pipeline.run('what do you know mahabharatha ?',[source(text,query_rel=.25)])
        self.assertEqual(len(result.claims),1)
        self.assertEqual(result.claims[0].text,text)
        self.assertEqual(query_overlap('mahabharatha','Mahābhārata'),1)

    def test_overview_lead_is_not_crowded_out_by_repeated_bibliographic_titles(self):
        lead=source('The Mahabharata is an ancient Sanskrit epic. The Mahabharata includes philosophical material.',query_rel=.11,url_or_provenance='https://same.example/article')
        bibliography=EvidencePassage('S2','A translation includes numerous books of the Mahabharata. The Mahabharata was adapted for television.',source_type='unknown',india_rel=0,query_rel=.35,source_signal='Unknown',language='en',url_or_provenance='https://same.example/books')
        result=self.pipeline.run('what do you know mahabharatha ?',[bibliography,lead])
        self.assertTrue(result.claims)
        self.assertTrue(result.claims[0].text.startswith('The Mahabharata is an ancient Sanskrit epic.'))

    def test_bracketed_pronunciation_is_preserved_as_exact_source_text(self):
        text='The Mahabharata (pronounced [maha]) is a Sanskrit epic from ancient India.'
        result=self.pipeline.run('What is Mahabharata?',[source(text)])
        self.assertEqual(result.claims[0].text,'The Mahabharata is a Sanskrit epic from ancient India.')
        self.assertEqual(result.claims[0].evidence[0].quote,text)
        self.assertEqual(result.claims[0].support_method,'pronunciation_metadata_omission')
        self.assertEqual(result.factors['evidence_coverage'],1)

    def test_query_scope_words_and_numbers_are_not_discarded(self):
        self.assertIn('first',query_terms('Who was the first president in 1950?'))
        self.assertIn('1950',query_terms('Who was the first president in 1950?'))
        self.assertIn('not',query_terms('What did the archive not permit?'))

    def test_emphatic_religious_authorship_is_not_detached_from_its_context(self):
        text='The actual author of the scripture is a deity. Some traditions give other accounts. Traditionally, the epic is attributed to a sage.'
        result=self.pipeline.run('Tell me about the epic scripture',[source(text)])
        self.assertNotIn('The actual author',result.answer)
        self.assertTrue(any('Traditionally' in c.text for c in result.claims))

    def test_compression_preserves_qualifiers_negation_and_dates(self):
        for text in ('The epic (according to one tradition) has eighteen books.',
                     'The name (pronounced differently only in one region) refers to an epic.',
                     'The book (published in 1985) includes a phonetic guide.'):
            self.assertEqual(concise_statement(text),text)

    def test_compressed_extraction_cannot_hide_an_unsupported_rewrite(self):
        p=source('The archive (pronounced [ark]) opened in 1985.')
        bad=GroundedClaim('The archive opened in 1990.',[ClaimEvidence(p.id,p.text)],'pronunciation_metadata_omission')
        rating,why,factors=assess([bad],[p],[p],[],{'status':'skipped'})
        self.assertEqual(factors['evidence_coverage'],0)
        self.assertEqual(rating.value,'LOW')

    def test_journal_page_ranges_and_related_article_promotions_are_not_answers(self):
        for text in ('"The Green Revolution in India: A Case Study in Technological Change," Technology and Culture 33 (October 1992): 737-56',
                     '" The Green Revolution of the 1960\'s and Its Impact on Small Farmers in India ", Environmental Studies Undergraduate Student Theses 10 (PDF)',
                     'The Violence of the green revolution : Third World agriculture, ecology, and politics .',
                     'The green revolution changed farming More From Encyclopedia Origins of agriculture The Green Revolution'):
            with self.subTest(text=text):
                result=self.pipeline.run('How did the Green Revolution change agriculture in India?',[source(text)])
                self.assertEqual(result.claims,[])


if __name__=='__main__':unittest.main()
