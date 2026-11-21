"""Regressions for context features, validated inputs and evidence metadata."""
from __future__ import annotations

import copy
from datetime import date

import pytest

import task4_rank_evidence as t4


def context(query, text):
    return t4.compute_indian_context(
        query, {"passage_id": "P", "text": text}, t4.DEFAULT_CONFIG["context_weights"]
    )["breakdown"]


@pytest.mark.parametrize("text", [
    "programming comparison digital basic",
    "telegram basis guitar",
    "रामायण",  # a primary text, not the separate person-name concept
])
def test_unrelated_substrings_do_not_match_entities(text):
    features = context("a general question", text)
    assert features["entity_feature"] == 0


def test_latin_substrings_do_not_match_other_lexicons():
    features = context("a general question", "programming comparison digital basic")
    assert features["institution_feature"] == 0
    assert features["primary_text_feature"] == 0


@pytest.mark.parametrize("text,feature", [
    ("Rama", "entity_feature"),
    ("Ram Rama राम", "entity_feature"),
    ("Vedas", "primary_text_feature"),
    ("Veda Vedas वेद", "primary_text_feature"),
    ("Ramayana", "primary_text_feature"),
    ("रामायण", "primary_text_feature"),
])
def test_aliases_count_canonical_concepts_once(text, feature):
    assert context("question", text)[feature] == 0.5


@pytest.mark.parametrize("query,text", [
    ("राम अयोध्या", "Rama lived in Ayodhya."),
    ("Rama Ayodhya", "राम का संबंध अयोध्या से है।"),
    ("Ram Rama राम Ayodhya अयोध्या", "Rama lived in Ayodhya."),
])
def test_query_context_matches_canonical_concepts_across_scripts(query, text):
    assert context(query, text)["query_context_match"] == 1.0


def test_script_ratio_uses_letters_without_numeral_or_mark_inflation():
    assert t4._devanagari_ratio("१२३ abc") == 0.0
    assert t4._devanagari_ratio("राम AB") == pytest.approx(0.5)
    assert 0 <= t4._devanagari_ratio("राम") <= 1


@pytest.mark.parametrize("query", [None, 1, "", " \n\t"])
def test_invalid_queries_are_rejected(query):
    with pytest.raises(ValueError):
        t4.validate_inputs(query, [])


@pytest.mark.parametrize("passages", [
    None,
    "not-a-list",
    [None],
    [1],
    ["passage_id"],
    [{}],
    [{"passage_id": ""}],
    [{"passage_id": " "}],
    [{"passage_id": 12}],
    [{"passage_id": "P"}, {"passage_id": "P"}],
    [{"passage_id": "P", "text": 12}],
    [{"passage_id": "P", "text": ["bad"]}],
    [{"passage_id": "P", "source_metadata": []}],
    [{"passage_id": "P", "extraction_metadata": "bad"}],
    [{"passage_id": "P", "source_metadata": {"publication_date": 2026}}],
    [{"passage_id": "P", "source_metadata": {"has_citations": "yes"}}],
    [{"passage_id": "P", "extraction_metadata": {"page_number": 0}}],
    [{"passage_id": "P", "extraction_metadata": {"page_number": True}}],
    [{"passage_id": "P", "extraction_metadata": {"method": 1}}],
])
def test_invalid_passages_are_rejected_before_encoder_load(passages, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid inputs must be rejected before encoder loading")
    monkeypatch.setattr(t4, "_get_semantic_model", unexpected)
    with pytest.raises(ValueError):
        t4.rank_evidence("question", passages)


@pytest.mark.parametrize("top_k", [-1, 1.2, "3", True])
def test_invalid_top_k_is_rejected_before_encoder_load(top_k, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid top_k must be rejected before encoder loading")
    monkeypatch.setattr(t4, "_get_semantic_model", unexpected)
    with pytest.raises(ValueError):
        t4.rank_evidence("question", [{"passage_id": "P", "text": "evidence"}], top_k=top_k)


@pytest.mark.parametrize("section", ["relevance_weights", "context_weights", "ranking_weights"])
@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), "invalid", None])
def test_all_weight_groups_are_validated_before_encoder_load(section, bad, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("bad weights must be rejected before encoder loading")
    monkeypatch.setattr(t4, "_get_semantic_model", unexpected)
    key = next(iter(t4.DEFAULT_CONFIG[section]))
    with pytest.raises(ValueError):
        t4.rank_evidence("question", [{"passage_id": "P", "text": "evidence"}],
                         config={section: {key: bad}})


@pytest.mark.parametrize("section", ["relevance_weights", "context_weights", "ranking_weights"])
def test_zero_weight_groups_are_rejected(section, monkeypatch):
    monkeypatch.setattr(t4, "_get_semantic_model",
                        lambda *args, **kwargs: pytest.fail("encoder loaded for invalid weights"))
    weights = dict.fromkeys(t4.DEFAULT_CONFIG[section], 0.0)
    with pytest.raises(ValueError):
        t4.rank_evidence("question", [{"passage_id": "P", "text": "evidence"}],
                         config={section: weights})


@pytest.mark.parametrize("section", ["context_weights", "ranking_weights"])
def test_scaling_weight_groups_preserves_ranking_and_scores(section, fake_semantic):
    passages = [
        {"passage_id": "P1", "text": "Rama in Ayodhya."},
        {"passage_id": "P2", "text": "An unrelated weather report."},
    ]
    baseline = t4.rank_evidence("Rama", passages)
    scaled = copy.deepcopy(t4.DEFAULT_CONFIG[section])
    scaled = {key: value * 10 for key, value in scaled.items()}
    assert t4.rank_evidence("Rama", passages, config={section: scaled}) == baseline


@pytest.mark.parametrize("confidence", [
    -0.1, 1.1, float("nan"), float("inf"), "unknown", True,
])
def test_invalid_ocr_confidence_is_rejected_before_model_load(confidence, monkeypatch):
    monkeypatch.setattr(t4, "_get_semantic_model",
                        lambda *args, **kwargs: pytest.fail("encoder loaded for invalid OCR confidence"))
    passage = {"passage_id": "P", "text": "extracted text",
               "extraction_metadata": {"ocr_confidence": confidence}}
    with pytest.raises(ValueError):
        t4.rank_evidence("question", [passage])


def test_fixed_reference_date_controls_recency():
    reference = "2026-11-09"
    expected = 1 - ((date(2026, 11, 9) - date(2023, 2, 10)).days / 365.25) / 30
    assert t4._recency_score("2023-02-10", reference_date=reference) == pytest.approx(expected)
    assert t4._recency_score(reference, reference_date=reference) == 1.0
    assert t4._recency_score("1900-01-01", reference_date=reference) == 0.5
    assert t4._recency_score(None, reference_date=reference) == 0.5
    assert t4._recency_score("bad-date", reference_date=reference) == 0.5


def test_evidence_keeps_ocr_and_context_separate():
    passage = {"passage_id": "P", "text": "राम अयोध्या",
               "source_metadata": {"source_type": "academic"},
               "extraction_metadata": {"ocr_confidence": 0.35}}
    evidence = t4.compute_source_signal(passage, 1.0, reference_date="2026-11-09")
    assert evidence["breakdown"]["ocr_score"] == 0.35
    assert 0 <= evidence["score"] <= 1
    international = dict(passage, text="A source written in English.")
    assert t4.compute_source_signal(international, 1.0, reference_date="2026-11-09") == evidence


def test_future_publication_date_is_neutral():
    assert t4._recency_score("2030-01-01", reference_date="2026-11-09") == 0.5


def test_missing_ocr_confidence_is_distinct_from_direct_extraction():
    scanned = {"passage_id": "S", "text": "scanned text",
               "extraction_metadata": {"method": "ocr"}}
    direct = {"passage_id": "D", "text": "direct text",
              "extraction_metadata": {"method": "direct"}}
    assert t4.compute_source_signal(scanned, 1, reference_date="2026-11-09")["breakdown"]["ocr_score"] == 0.5
    assert t4.compute_source_signal(direct, 1, reference_date="2026-11-09")["breakdown"]["ocr_score"] == 1.0


@pytest.mark.parametrize("joiner", ["\u200c", "\u200d"])
def test_hindi_conjunct_joiners_preserve_alias_and_lexical_matches(joiner):
    joined = "कृष्" + joiner + "ण"
    assert context("Krishna", joined)["query_context_match"] == 1.0
    assert context("question", joined)["entity_feature"] == 0.5
    score = t4.lexical_relevance("कृष्ण", [{"passage_id": "K", "text": joined}])
    assert score[0] == pytest.approx(1.0)


@pytest.mark.parametrize("url", [
    "https://[broken",
    "https://[invalid]/path",
    "https://example.org:bad",
    "https://:443",
    "https://exa mple.org",
])
def test_malformed_urls_are_neutral_instead_of_crashing(url):
    passage = {"passage_id": "URL", "text": "evidence",
               "source_metadata": {"url": url}}
    config = {"relevance_weights": {"lexical_weight": 1.0, "semantic_weight": 0.0}}
    ranked = t4.rank_evidence("question", [passage], config=config)
    assert ranked[0]["feature_breakdown"]["evidence"]["provenance_score"] == 0.4


def test_valid_source_url_allows_surrounding_whitespace():
    passage = {"passage_id": "URL", "text": "evidence",
               "source_metadata": {"url": "  https://example.org:443/paper  "}}
    score = t4.compute_source_signal(passage, 1, reference_date="2026-11-09")
    assert score["breakdown"]["provenance_score"] == 1.0


@pytest.mark.parametrize("metadata,expected", [
    ({"author": "   "}, 0.3),
    ({"organization": "\n\t"}, 0.3),
    ({"author": " ", "organization": "  "}, 0.3),
    ({"author": " Example author "}, 0.7),
    ({"organization": " Example institute "}, 0.7),
])
def test_only_nonblank_attribution_receives_credit(metadata, expected):
    passage = {"passage_id": "P", "text": "evidence", "source_metadata": metadata}
    score = t4.compute_source_signal(passage, 1, reference_date="2026-11-09")
    assert score["breakdown"]["attribution_score"] == expected


@pytest.mark.parametrize("override", [
    {"evidence": {"reference_date": "2020-01-01"}},
    {"evidence": {"duplicate_threshold": 0.5}},
    {"semantic_model": {"name": "another-model"}},
    {"semantic_model": {"revision": "another-revision"}},
])
def test_cached_features_reject_changed_extraction_settings(fake_semantic, override):
    config = {"evidence": {"reference_date": "2026-11-09"}}
    passage = {"passage_id": "P", "text": "Rama in Ayodhya",
               "source_metadata": {"publication_date": "2023-01-01"}}
    prepared = t4.prepare_features("Rama", [passage], config=config)
    changed = t4.merge_config(config, override)
    with pytest.raises(ValueError, match="extraction settings"):
        t4.rank_prepared(prepared, config=changed)


def test_cached_semantic_features_allow_weight_filter_and_output_changes(fake_semantic):
    config = {"evidence": {"reference_date": "2026-11-09"}}
    passage = {"passage_id": "P", "text": "Rama in Ayodhya"}
    prepared = t4.prepare_features("Rama", [passage], config=config)
    changed = t4.merge_config(config, {
        "relevance_weights": {"lexical_weight": 1.0, "semantic_weight": 0.0},
        "ranking_weights": {"query_relevance_weight": 1.0,
                            "indian_context_weight": 0.0, "evidence_weight": 0.0},
        "output": {"include_provenance": False},
        "filter": {"enabled": True, "threshold": 0.0},
    })
    ranked = t4.rank_prepared(prepared, config=changed)
    assert ranked[0]["passage_id"] == "P"
    assert "provenance" not in ranked[0]


def test_prepared_signature_resolves_implicit_reference_date(fake_semantic):
    prepared = t4.prepare_features("Rama", [{"passage_id": "P", "text": "Rama"}])
    signature = prepared[0]["extraction_config"]
    assert signature["schema_version"] == 2
    assert signature["reference_date"] == date.today().isoformat()


def test_default_bge_model_prefers_installed_local_directory(monkeypatch):
    import sys
    import types

    constructors = []
    model = object()
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = lambda *args, **kwargs: constructors.append((args, kwargs)) or model
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setattr(t4, "_SEMANTIC_MODEL", None)
    monkeypatch.setattr(t4, "_SEMANTIC_MODEL_KEY", None)
    monkeypatch.setattr(t4, "_local_semantic_model_directory", lambda: "test-local-bge-model")
    result = t4._get_semantic_model({"local_files_only": True, "revision": t4._LOCAL_BGE_REVISION})
    assert result is model
    assert constructors[0][0] == ("test-local-bge-model",)
    assert constructors[0][1]["local_files_only"] is True
    assert "revision" not in constructors[0][1]
    assert t4._SEMANTIC_MODEL_KEY[:3] == ("BAAI/bge-m3", None, t4._LOCAL_BGE_REVISION)


def test_explicit_other_model_revision_is_not_replaced_by_local_snapshot(monkeypatch):
    import sys
    import types

    constructors = []
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = lambda *args, **kwargs: constructors.append((args, kwargs)) or object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setattr(t4, "_SEMANTIC_MODEL", None)
    monkeypatch.setattr(t4, "_SEMANTIC_MODEL_KEY", None)
    monkeypatch.setattr(t4, "_local_semantic_model_directory", lambda: "test-local-bge-model")
    t4._get_semantic_model({"revision": "some-other-revision", "local_files_only": True})
    assert constructors[0][0] == ("BAAI/bge-m3",)
    assert constructors[0][1]["revision"] == "some-other-revision"
