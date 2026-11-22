"""Independent contract expectations, not imports of the upstream implementation."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from task5.config import Task5Config
from task5.handoff import adapt_task4_envelope, process_task4_output, summarize_task4_payload


def envelope(query="How does the archive help researchers?", rows=None):
    if rows is None:
        rows = [dict(rank=1, passage_id="original doc/page 1", text="The archive helps researchers find public catalogue records.",
            query_relevance=.9, indian_context_relevance=.6, source_signal=.73, final_score=.8,
            provenance={"source_metadata":{"document_id":"D1", "source_type":"academic",
              "url":"https://archive.example.org/catalogue", "title":"Archive catalogue", "language":"en"},
              "extraction_metadata":{"method":"DIRECT_EXTRACTION", "page_number":1}})]
    refs=[]
    for row in rows:
        provenance=row.get("provenance",{})
        source,extraction=provenance.get("source_metadata",{}),provenance.get("extraction_metadata",{})
        reference={"passage_id":row["passage_id"],"reference_available":bool(source.get("url") or source.get("document_id") or source.get("source_id"))}
        for key in ("document_id","source_id","url","title","language","source_group_id","origin_document_id"):
            if key in source:reference[key]=source[key]
        for key in ("method","page_number","ocr_confidence","image_id","page_id"):
            if key in extraction:reference[key]=extraction[key]
        refs.append(reference)
    return dict(schema_version="1.0.0",query=query,status="ok" if rows else "no_relevant_evidence",ranked_evidence=rows,
        is_simulated=False,provenance={"kind":"task3_output","description":"Supplied extraction output"},
        traceability={"selected_evidence_ids":[r["passage_id"] for r in rows],"selected_references":refs,
          "references_available":bool(refs) and all(r["reference_available"] for r in refs),
          "source_support_scope":"all_input_candidates_before_filtering" if any("source_support" in r for r in rows) else "not_included",
          "source_independence_established":False,"text_similarity_establishes_agreement":False},
        integration={"task5":"consumer_stub_only","actual_task5_available":False})


class HandoffTests(unittest.TestCase):
    def setUp(self):self.config=Task5Config(provider="extractive",min_query_rel=.10)

    def test_numeric_signals_and_nested_provenance_preserved(self):
        raw=envelope();before=copy.deepcopy(raw)
        passages,mapping=adapt_task4_envelope(raw)
        p=passages[0]
        self.assertEqual(p.query_rel,.9)
        self.assertEqual(p.india_rel,.6)
        self.assertEqual(p.metadata["task4"]["source_signal"],.73)
        self.assertEqual(p.source_signal,"High")
        self.assertEqual(p.url_or_provenance,"https://archive.example.org/catalogue")
        self.assertEqual(p.page,1)
        self.assertEqual(mapping,{"S1":"original doc/page 1"})
        self.assertEqual(raw,before)

    def test_normalized_ocr_is_not_divided_again(self):
        raw=envelope();row=raw["ranked_evidence"][0]
        row["provenance"]["extraction_metadata"]={"method":"TESSERACT_OCR","ocr_confidence":.93,"page_number":2}
        raw=envelope(rows=[row]);p=adapt_task4_envelope(raw)[0][0]
        self.assertEqual(p.ocr_conf,.93);self.assertEqual(p.page,2)

    def test_direct_text_conventional_confidence_not_treated_as_ocr(self):
        raw=envelope();row=raw["ranked_evidence"][0]
        row["provenance"]["extraction_metadata"]["ocr_confidence"]=1.
        self.assertIsNone(adapt_task4_envelope(envelope(rows=[row]))[0][0].ocr_conf)

    def test_bad_ocr_scale_rejected(self):
        raw=envelope();row=raw["ranked_evidence"][0]
        row["provenance"]["extraction_metadata"]={"method":"OCR","ocr_confidence":93}
        with self.assertRaises(ValueError):adapt_task4_envelope(envelope(rows=[row]))

    def test_different_document_ids_same_publisher_are_one_group(self):
        a=envelope()["ranked_evidence"][0];b=copy.deepcopy(a)
        b.update(rank=2,passage_id="B",text="Staff help researchers request rare manuscript records.")
        b["provenance"]["source_metadata"].update(document_id="D2",url="https://news.example.org/d2")
        p,_=adapt_task4_envelope(envelope(rows=[a,b]))
        self.assertEqual(p[0].source_identity(),p[1].source_identity())

    def test_declared_mirrors_across_domains_stay_grouped(self):
        a=envelope()["ranked_evidence"][0];b=copy.deepcopy(a)
        a["provenance"]["source_metadata"]["origin_document_id"]="O"
        b.update(rank=2,passage_id="B")
        b["provenance"]["source_metadata"].update(url="https://other.org/mirror",origin_document_id="O")
        p,_=adapt_task4_envelope(envelope(rows=[a,b]))
        self.assertEqual(p[0].source_identity(),p[1].source_identity())

    def test_numeric_composite_never_turns_unknown_source_into_high(self):
        row=envelope()["ranked_evidence"][0];row["source_signal"]=1.
        del row["provenance"]["source_metadata"]["source_type"]
        self.assertEqual(adapt_task4_envelope(envelope(rows=[row]))[0][0].source_signal,"Unknown")

    def test_duplicate_ids_rank_and_status_rejected(self):
        for mutation in (lambda p:p.update(status="no_relevant_evidence"),
                         lambda p:p["ranked_evidence"][0].update(rank=True),
                         lambda p:p["traceability"].update(selected_evidence_ids=["made-up"]),
                         lambda p:p["ranked_evidence"][0].update(query_relevance=float("nan"))):
            p=envelope();mutation(p)
            with self.assertRaises(ValueError):adapt_task4_envelope(p)

    def test_conflicting_traceability_alias_is_rejected(self):
        p=envelope();p["traceability"]["selected_references"][0]["url"]="https://wrong.org"
        with self.assertRaises(ValueError):adapt_task4_envelope(p)

    def test_support_counters_are_not_corroboration(self):
        row=envelope()["ranked_evidence"][0];row["source_support"]={"similar_text_source_group_count":100}
        result=summarize_task4_payload(envelope(rows=[row]),config=self.config)["task5_output"]["queries"][0]["result"]
        self.assertEqual(result["factors"]["cross_source_agreement"],0.)
        self.assertNotEqual(result["reliability"],"HIGH")

    def test_multiquery_isolation_and_empty_query_evidence(self):
        a=envelope();b=envelope("A different question?",[])
        out=summarize_task4_payload({"task4_output":{"schema_version":"1.0","queries":[a,b],"metadata":{"original_query":[a['query'],b['query']]}}},config=self.config)["task5_output"]
        self.assertEqual(out["summary"],{"questions":2,"answered":1,"abstained":1})
        self.assertEqual(out["queries"][1]["result"]["cited_sources"],[])
        self.assertEqual(out["queries"][0]["integration"]["actual_task5_available"],True)

    def test_simulated_evidence_stays_labelled(self):
        p=envelope();p["is_simulated"]=True;p["provenance"]["kind"]="constructed_example"
        self.assertTrue(adapt_task4_envelope(p)[0][0].metadata["synthetic"])

    def test_partial_upstream_failure_is_reported(self):
        p=envelope();p["provenance"]["document_status_counts"]={"failed":1}
        r=summarize_task4_payload(p,config=self.config)["task5_output"]["queries"][0]["result"]
        self.assertTrue(any("failed extraction" in w for w in r["warnings"]))

    def test_output_errors_preserve_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"task4.json";dest=Path(tmp)/"answer.json"
            source.write_text(json.dumps({"invalid":True}));dest.write_text("previous valid result")
            with self.assertRaises(ValueError):process_task4_output(source,dest,config=self.config)
            self.assertEqual(dest.read_text(),"previous valid result")

    def test_input_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"same.json";path.write_text(json.dumps(envelope()))
            with self.assertRaises(ValueError):process_task4_output(path,path,config=self.config)

    def test_copied_document_is_grouped_with_its_original(self):
        a=envelope()["ranked_evidence"][0];b=copy.deepcopy(a)
        b.update(rank=2,passage_id="copy")
        b["provenance"]["source_metadata"].update(document_id="COPY",url="https://other.org/copy",origin_document_id="D1")
        p,_=adapt_task4_envelope(envelope(rows=[a,b]))
        self.assertEqual(p[0].source_identity(),p[1].source_identity())

    def test_question_headers_are_not_generated_as_answer_claims(self):
        p=envelope();p["ranked_evidence"][0]["text"]="What is an archive? The archive provides researchers with public catalogue records. Archive catalogue"
        r=summarize_task4_payload(p,config=self.config)["task5_output"]["queries"][0]["result"]
        self.assertTrue(r["claims"])
        self.assertTrue(all("?" not in c["text"] for c in r["claims"]))
        self.assertNotIn("Archive catalogue [",r["answer"])

    def test_task3_unpunctuated_body_text_remains_usable(self):
        p=envelope();p["ranked_evidence"][0]["text"]="The Green Revolution increased wheat production in Punjab through new seeds"
        r=summarize_task4_payload(p,config=self.config)["task5_output"]["queries"][0]["result"]
        self.assertTrue(r["claims"])
        self.assertEqual(r["claims"][0]["evidence"][0]["quote"],p["ranked_evidence"][0]["text"])


if __name__=="__main__":unittest.main()
