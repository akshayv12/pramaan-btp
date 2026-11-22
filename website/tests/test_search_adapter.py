"""Regression for the necessary upstream DDGS URL-validity fix."""
import importlib.util
from unittest.mock import patch
import unittest

from website.backend import ROOT


class SearchAdapterTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("website_search_adapter_test", ROOT / "task2/search_engines.py")
        self.module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.module)
        except ImportError:
            self.skipTest("Use the installed pipeline Python environment for this adapter regression")

    def test_invalid_provider_paths_allow_existing_bing_fallback(self):
        calls = []

        class FakeDDGS:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def text(self, query, **kwargs):
                calls.append(kwargs["backend"])
                if kwargs["backend"] == "auto":
                    return [{"href": "/go/v1/opaque-redirect"}, {"href": "javascript:bad"}, {"href": None}, {"href": "https://[bad"}]
                return [{"title": "Useful source", "href": "https://example.org/evidence", "body": "Passage"}]

        with patch.object(self.module, "DDGS", FakeDDGS):
            results, engine = self.module.search_ddgs("India agriculture")
        self.assertEqual(calls, ["auto", "bing"])
        self.assertEqual(engine, "ddgs-bing")
        self.assertEqual(results[0]["url"], "https://example.org/evidence")

    def test_valid_auto_results_keep_the_existing_pipeline_order(self):
        calls = []

        class FakeDDGS:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def text(self, query, **kwargs):
                calls.append(kwargs["backend"])
                return [{"title": "Source", "href": "https://example.org/", "body": "Text"}]

        with patch.object(self.module, "DDGS", FakeDDGS):
            results, engine = self.module.search_ddgs("question")
        self.assertEqual(calls, ["auto"])
        self.assertEqual(engine, "ddgs-auto")
        self.assertEqual(results, [{"title": "Source", "url": "https://example.org/", "text": "Text"}])


if __name__ == "__main__":
    unittest.main()
