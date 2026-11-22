import pytest

from app import generate_queries, generate_queries_brief, to_task2_response


@pytest.fixture(autouse=True)
def isolate_backend(monkeypatch):
    monkeypatch.setenv("QUERY_BACKEND", "baseline")
    monkeypatch.setenv("TRANSLATION_PROVIDER", "rules")


@pytest.mark.parametrize(
    ("query", "expected_language"),
    [
        ("राम कैसे थे राम को क्यों भगवान मानते हैं लोग?", "hindi_devanagari"),
        ("Ram kaise the?", "roman_hindi"),
        ("Green Revolution impact on Indian farmers?", "english"),
    ],
)
def test_language_detection(query, expected_language):
    result = generate_queries(query, 8)
    assert result.detected_language == expected_language


def test_brief_output_shape():
    output = generate_queries_brief("Ram kaise the?", 8)
    assert set(output) == {
        "language",
        "normalized_query",
        "search_queries",
        "key_terms",
    }


def test_query_budget_is_validated_for_direct_callers():
    with pytest.raises(ValueError, match="between 2 and 10"):
        generate_queries("What is ISRO?", 1)


def test_task2_handoff_preserves_ranked_query_shape():
    result = to_task2_response(generate_queries("Green Revolution impact on Indian farmers?", 4))

    assert result.queries
    assert [query.rank for query in result.queries] == list(range(1, len(result.queries) + 1))
    assert all(query.text and query.language in {"hi", "en", "roman_hindi"} for query in result.queries)


def test_deduplication_ignores_case_whitespace_and_trailing_punctuation():
    result = generate_queries(
        "How did the Green Revolution affect small farmers in India?",
        4,
    )
    texts = [item.query.strip().casefold().rstrip("?.!।") for item in result.search_queries]

    assert len(texts) == len(set(texts))
