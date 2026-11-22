"""Source-aware grouping and conservative reviewed-support contracts."""
import copy
import numpy as np
import pytest
import task4_rank_evidence as t4
from evidence_groups import analyze_source_support, canonical_source_url

def passage(pid, text="The lunar sample contains sulphur.", **metadata):
    return {"passage_id": pid, "text": text, "source_metadata": metadata}

def test_url_identity_removes_tracking_preserves_meaning():
    assert canonical_source_url("https://EXAMPLE.org/report?id=2&utm_source=a#page") == "https://example.org/report?id=2"
    assert canonical_source_url("https://example.org/report?id=3") != canonical_source_url("https://example.org/report?id=2")

@pytest.mark.parametrize("url", ["https://[broken", "https://example.org:bad", "javascript:alert(1)",
                                 "https://user:password@example.org", "https://example.org:0"])
def test_bad_url_not_a_source_identity(url):
    assert canonical_source_url(url) is None

def test_same_document_tracking_links_collapse():
    ps = [passage("A", url="https://example.org/report?utm_source=x"),
          passage("B", url="https://example.org/report?utm_source=y")]
    groups = analyze_source_support(ps, np.ones((2, 2)))
    assert groups[0]["document_id"] == groups[1]["document_id"]
    assert groups[0]["same_origin_passage_ids"] == ["B"]

def test_declared_copies_share_origin_without_claiming_independence():
    ps = [passage("A", document_id="original", organization="Original Lab"),
          passage("B", document_id="copy", origin_document_id="original", organization="News Site")]
    groups = analyze_source_support(ps, np.ones((2, 2)))
    assert groups[0]["origin_group_id"] == groups[1]["origin_group_id"]
    assert groups[0]["near_duplicate_passage_ids"] == ["B"]
    assert groups[0]["source_independence_established"] is False
    assert groups[0]["text_similarity_establishes_agreement"] is False
    assert groups[0]["reviewed_claims"] == []

def test_different_domains_are_only_similarity_candidates():
    ps = [passage("A", url="https://one.example/report"),
          passage("B", url="https://two.example/report")]
    group = analyze_source_support(ps, np.ones((2, 2)))[0]
    assert group["similar_text_source_group_count"] == 2
    assert group["text_similarity_establishes_agreement"] is False

def annotation(stance="supports"):
    return {"claim_id": "C1", "reviewer_id": "unit-test-reviewer",
            "review_status": "reviewed", "stance": stance,
            "supporting_span": "contains sulphur"}

def test_reviewed_stances_keep_disagreement_visible():
    ps = [passage("A", document_id="A", organization="Lab A", support_annotations=[annotation()]),
          passage("B", document_id="B", organization="Lab B", support_annotations=[annotation("contradicts")])]
    record = analyze_source_support(ps, np.ones((2, 2)))[0]["reviewed_claims"][0]
    assert record["supporting_document_origins"] == 1
    assert record["contradicting_document_origins"] == 1
    assert record["has_reviewed_disagreement"] is True

def test_duplicate_review_declarations_count_one_origin():
    ps = [passage("A", document_id="A", support_annotations=[annotation()]),
          passage("B", document_id="B", origin_document_id="A", support_annotations=[annotation()])]
    record = analyze_source_support(ps, np.ones((2, 2)))[0]["reviewed_claims"][0]
    assert record["supporting_document_origins"] == 1
    assert record["supporting_source_groups"] == 0

def test_pending_support_is_not_reviewed_corroboration():
    ps = [passage("A", support_annotations=[{"review_status": "pending"}])]
    assert analyze_source_support(ps, np.eye(1))[0]["reviewed_claims"] == []

def test_bad_support_declaration_fails_before_model(monkeypatch):
    record = annotation()
    record["supporting_span"] = "text not present"
    monkeypatch.setattr(t4, "_get_semantic_model",
                        lambda *a, **k: pytest.fail("model loaded for invalid metadata"))
    with pytest.raises(ValueError, match="supporting_span"):
        t4.rank_evidence("Moon composition", [passage("A", support_annotations=[record])])

def test_source_support_opt_in_keeps_default_contract(fake_semantic):
    ps = [passage("A", document_id="A"), passage("B", document_id="B", origin_document_id="A")]
    base = t4.rank_evidence("Moon composition", ps)
    enabled = t4.rank_evidence("Moon composition", ps,
                              config={"output": {"include_source_support": True}})
    assert "source_support" not in base[0]
    assert enabled[0]["source_support"]["scope"] == "all_input_candidates_before_filtering"
    assert [{k:v for k,v in item.items() if k!="source_support"} for item in enabled] == base

def test_support_indicators_do_not_reward_similar_text(fake_semantic):
    ps = [passage("A", document_id="A"), passage("B", document_id="B")]
    plain = t4.rank_evidence("Moon composition", ps)
    revised = copy.deepcopy(ps)
    revised[0]["source_metadata"]["support_annotations"] = [annotation()]
    annotated = t4.rank_evidence("Moon composition", revised)
    assert [item["final_score"] for item in plain] == [item["final_score"] for item in annotated]

def test_low_duplicate_threshold_partial_override_remains_valid():
    cfg = t4.validated_config({"evidence": {"duplicate_threshold": 0.5}})
    assert cfg["evidence"]["related_text_threshold"] <= 0.5

def test_repeated_query_parameter_order_not_erased():
    assert canonical_source_url("https://example.org/?mode=a&mode=b") != canonical_source_url("https://example.org/?mode=b&mode=a")

def test_copied_organization_does_not_add_reviewed_source_group():
    ps = [passage("A", document_id="A", organization="Original Lab", support_annotations=[annotation()]),
          passage("B", document_id="B", origin_document_id="A", organization="Republisher", support_annotations=[annotation()])]
    record = analyze_source_support(ps, np.ones((2,2)))[0]["reviewed_claims"][0]
    assert record["supporting_document_origins"] == 1
    assert record["supporting_source_groups"] == 1

def test_conflicting_copy_origin_aliases_rejected():
    with pytest.raises(ValueError, match="conflict"):
        t4.validate_inputs("query", [passage("A", origin_document_id="one", copied_from_document_id="two")])
