"""Targeted tests for the necessary fixes in the existing extraction and ranking routes."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from website.backend import ROOT

sys.path.insert(0,str(ROOT/'task4'))


class UpstreamQualityTests(unittest.TestCase):
    def test_topic_spelling_beats_matching_conversational_quiz_words(self):
        from task4_rank_evidence import lexical_relevance
        passages=[{'text':'The Mahābhārata is an ancient Sanskrit epic from India.'},
                  {'text':'How well do you know the Bible? Thank you for your feedback!'}]
        before=copy.deepcopy(passages)
        scores=lexical_relevance('what do you know mahabharatha ?',passages)
        self.assertGreater(scores[0],scores[1])
        self.assertGreater(scores[0],0)
        self.assertEqual(scores[1],0)
        self.assertEqual(passages,before)

    def test_fuzzy_matching_never_rewrites_numbers(self):
        from task4_rank_evidence import lexical_relevance
        self.assertEqual(lexical_relevance('1985',[{'text':'1990'}])[0],0)

    def test_article_extraction_ignores_empty_parser_node_and_keeps_inline_words(self):
        path=ROOT/'task3/src/extractor.py'
        spec=importlib.util.spec_from_file_location('quality_extractor_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        html='''<html><body><nav>Menu unrelated terms</nav><div class="mw-parser-output"></div>
        <main><div class="mw-parser-output"><table class="navbox"><tr><td>Rama Krishna Navigation</td></tr></table>
        <p>The <b>Mahābhārata</b> is an ancient <a>Sanskrit</a> epic.</p>
        <p>It includes philosophical material.</p><form>Thank you for your feedback!</form></div></main>
        <footer>Privacy and cookies</footer></body></html>'''
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'article.html';file.write_text(html)
            text=module.extract_text_direct(str(file))
        self.assertIn('The Mahābhārata is an ancient Sanskrit epic.',text)
        for noise in ('Navigation','feedback','Privacy','Menu'):
            self.assertNotIn(noise,text)


if __name__=='__main__':unittest.main()
