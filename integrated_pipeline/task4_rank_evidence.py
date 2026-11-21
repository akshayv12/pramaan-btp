"""Indian-context evidence prioritization.

rank_evidence(query, passages, top_k=None, config=None) returns interpretable
passage rankings. Context and source evidence are separate signals.
The BGE-M3 encoder is loaded lazily; importing this module never downloads it.
prepare_features/rank_prepared separate expensive extraction from weight tuning.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import re
import sys
import unicodedata
from difflib import get_close_matches
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from evidence_groups import analyze_source_support, validate_support_metadata
from context_features import (
    CONTEXT_ALIASES, normalize_text, unicode_tokens, matched_concepts,
    devanagari_ratio, extract_context_features,
)

FEATURE_LEXICONS = {category: {alias for aliases in concepts.values() for alias in aliases}
                    for category, concepts in CONTEXT_ALIASES.items()}
INDIAN_ENTITIES = FEATURE_LEXICONS["entity_feature"]
INDIAN_PLACES = FEATURE_LEXICONS["place_feature"]
INDIAN_INSTITUTIONS = FEATURE_LEXICONS["institution_feature"]
PRIMARY_TEXTS = FEATURE_LEXICONS["primary_text_feature"]
CONTEXTUAL_VOCAB = FEATURE_LEXICONS["contextual_vocab_feature"]
CONTEXT_WEIGHT_KEYS = {
    "script_feature": "w_script", "entity_feature": "w_entity",
    "place_feature": "w_place", "institution_feature": "w_institution",
    "primary_text_feature": "w_primary", "contextual_vocab_feature": "w_vocab",
    "query_context_match": "w_query_ctx",
}
SOURCE_TYPE_PRIOR = {
    "academic": 0.95, "institutional": 0.9, "scholarly": 0.9,
    "book": 0.85, "manuscript": 0.8, "news": 0.6, "blog": 0.35,
    "anonymous": 0.15, None: 0.4,
}
DEFAULT_CONFIG = {
    "relevance_weights": {"lexical_weight": 0.30, "semantic_weight": 0.70},
    "semantic_model": {"name": "BAAI/bge-m3", "batch_size": 16, "device": None},
    "context_weights": {
        "w_script": 0.15, "w_entity": 0.20, "w_place": 0.15,
        "w_institution": 0.15, "w_primary": 0.15, "w_vocab": 0.10,
        "w_query_ctx": 0.10,
    },
    "ranking_weights": {
        "query_relevance_weight": 0.109, "indian_context_weight": 0.349,
        "evidence_weight": 0.543,
    },
    "filter": {"enabled": False, "threshold": 0.0},
    "evidence": {"reference_date": None, "duplicate_threshold": 0.92, "related_text_threshold": 0.65},
    "output": {"include_feature_breakdown": True, "include_provenance": True, "include_source_support": False},
}

def merge_config(base: dict, override: Optional[dict]) -> dict:
    if override is not None and not isinstance(override, dict):
        raise ValueError("config must be a dict")
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def _finite_number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result

def _clip01(value: float) -> float:
    return max(0.0, min(1.0, _finite_number(value, "score")))

def _normalize_weights(weights: dict, keys, name: str) -> dict:
    if not isinstance(weights, dict):
        raise ValueError(f"{name} must be a dict")
    if set(weights) != set(keys):
        raise ValueError(f"{name} must contain exactly {list(keys)}")
    result = {key: _finite_number(weights[key], f"{name}.{key}") for key in keys}
    if any(value < 0 for value in result.values()):
        raise ValueError(f"{name} must be non-negative")
    # Scale before summing to avoid overflow for large but finite values.
    scale = max(result.values())
    if scale <= 0:
        raise ValueError(f"{name} must sum to more than zero")
    scaled = {key: value / scale for key, value in result.items()}
    total = sum(scaled.values())
    return {key: value / total for key, value in scaled.items()}

def validate_relevance_weights(weights: dict) -> dict:
    return _normalize_weights(weights, DEFAULT_CONFIG["relevance_weights"], "relevance_weights")

def _reference_date(value=None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError("reference_date must be an ISO date YYYY-MM-DD")

def validate_inputs(query: str, passages: list[dict]) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(passages, list):
        raise ValueError("passages must be a list of dicts")
    seen = set()
    for passage in passages:
        if not isinstance(passage, dict):
            raise ValueError("every passage must be a dict")
        pid = passage.get("passage_id")
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError("every passage needs a non-empty string passage_id")
        if pid in seen:
            raise ValueError(f"duplicate passage_id: {pid}")
        seen.add(pid)
        if passage.get("text") is not None and not isinstance(passage["text"], str):
            raise ValueError("passage text must be a string or None")
        for section in ("source_metadata", "extraction_metadata"):
            if passage.get(section) is not None and not isinstance(passage[section], dict):
                raise ValueError(f"{section} must be a dict or None")
        meta = passage.get("source_metadata") or {}
        for field in ("source_type", "url", "author", "organization", "publication_date"):
            if meta.get(field) is not None and not isinstance(meta[field], str):
                raise ValueError(f"source_metadata.{field} must be a string or None")
        if meta.get("has_citations") is not None and not isinstance(meta["has_citations"], bool):
            raise ValueError("source_metadata.has_citations must be a boolean")
        extraction = passage.get("extraction_metadata") or {}
        if extraction.get("ocr_confidence") is not None:
            confidence = _finite_number(extraction["ocr_confidence"], "ocr_confidence")
            if not 0 <= confidence <= 1:
                raise ValueError("ocr_confidence must be in [0, 1]")
        page = extraction.get("page_number")
        if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
            raise ValueError("page_number must be a positive integer")
        if extraction.get("method") is not None and not isinstance(extraction["method"], str):
            raise ValueError("extraction_metadata.method must be a string or None")
        validate_support_metadata(passage)

def _validate_top_k(top_k) -> None:
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
        raise ValueError("top_k must be a non-negative integer or None")

def validated_config(config=None) -> dict:
    cfg = merge_config(DEFAULT_CONFIG, config)
    for section in ("semantic_model", "filter", "evidence", "output"):
        if not isinstance(cfg[section], dict):
            raise ValueError(f"{section} must be a dict")
    cfg["relevance_weights"] = validate_relevance_weights(cfg["relevance_weights"])
    for section in ("context_weights", "ranking_weights"):
        cfg[section] = _normalize_weights(cfg[section], DEFAULT_CONFIG[section], section)
    if not isinstance(cfg["filter"]["enabled"], bool):
        raise ValueError("filter.enabled must be a boolean")
    threshold = _finite_number(cfg["filter"]["threshold"], "filter.threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("filter.threshold must be in [0, 1]")
    cfg["filter"]["threshold"] = threshold
    duplicate_threshold = _finite_number(cfg["evidence"]["duplicate_threshold"], "duplicate_threshold")
    if not 0 <= duplicate_threshold <= 1:
        raise ValueError("duplicate_threshold must be in [0, 1]")
    cfg["evidence"]["duplicate_threshold"] = duplicate_threshold
    if "related_text_threshold" not in ((config or {}).get("evidence") or {}):
        cfg["evidence"]["related_text_threshold"] = min(cfg["evidence"]["related_text_threshold"], duplicate_threshold)
    related = _finite_number(cfg["evidence"]["related_text_threshold"], "related_text_threshold")
    if not 0 <= related <= duplicate_threshold:
        raise ValueError("related_text_threshold must be in [0, duplicate_threshold]")
    cfg["evidence"]["related_text_threshold"] = related
    if cfg["evidence"]["reference_date"] is not None:
        cfg["evidence"]["reference_date"] = _reference_date(cfg["evidence"]["reference_date"]).isoformat()
    for option in ("include_feature_breakdown", "include_provenance", "include_source_support"):
        if not isinstance(cfg["output"][option], bool):
            raise ValueError(f"output.{option} must be a boolean")
    _semantic_model_config(cfg["semantic_model"])
    return cfg

def _text_of(passage: dict) -> str:
    return (passage.get("text") or "").strip()

def _tfidf_cosine(query: str, texts: list[str], analyzer="word", ngram_range=(1, 1)) -> np.ndarray:
    scores = np.zeros(len(texts), dtype=float)
    if not any(text.strip() for text in texts):
        return scores
    options = {"analyzer": analyzer, "ngram_range": ngram_range, "min_df": 1}
    if analyzer == "word":
        options.update(tokenizer=unicode_tokens, token_pattern=None, lowercase=False)
    try:
        matrix = TfidfVectorizer(**options).fit_transform([query] + texts)
        scores = cosine_similarity(matrix[:1], matrix[1:]).ravel()
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
    return np.clip(scores, 0, 1)

def lexical_relevance(query: str, passages: list[dict]) -> np.ndarray:
    """Word TF-IDF on the topic, with bounded corpus-based spelling variants.

    Conversational scaffolding must not match quiz controls such as 'you know'.
    Latin diacritics are folded for matching; Devanagari marks are retained.
    The original query, passage text and all provenance remain unchanged.
    """
    def fold(text):
        return re.sub(r'[\u0300-\u036f]', '', unicodedata.normalize('NFKD', text)).casefold()
    filler = set('what which who when where why how do does did you your we our i me my know tell about please explain describe give overview introduction the a an of in on for to is are was were can could would'.split())
    texts = [fold(_text_of(passage)) for passage in passages]
    terms = [term for term in unicode_tokens(fold(query)) if term not in filler]
    if not terms:
        return np.zeros(len(passages), dtype=float)
    vocabulary = sorted({word for text in texts for word in unicode_tokens(text) if word.isascii() and word.isalpha() and len(word) >= 6})
    terms = list(dict.fromkeys(terms))
    variants = [' '.join(terms)]
    for index, term in enumerate(terms[:20]):
        if not term.isascii() or not term.isalpha() or len(term) < 6:
            continue
        # At most two close variants per term; never fuzz numbers or short names.
        matches = get_close_matches(term, vocabulary, n=3, cutoff=.90)
        for word in matches:
            if word[:3] == term[:3] and word != term and len(variants)<4:
                alternate = terms.copy(); alternate[index] = word
                variants.append(' '.join(alternate))
    # Score alternatives separately: appending an unmatched spelling would dilute
    # a one-topic query and suppress the actual article. No alternative adds facts.
    return np.maximum.reduce([_tfidf_cosine(variant, texts) for variant in dict.fromkeys(variants)])

class SemanticModelUnavailableError(RuntimeError):
    """The required multilingual semantic encoder could not be loaded."""

_MODEL_HELP = """Could not load the multilingual semantic encoder {name!r}.
Reason: {reason}
Install sentence-transformers and PyTorch: pip install -r requirements.txt
Default model: BAAI/bge-m3 (about 2.2 GB of model weights).
Download it once with SentenceTransformer('BAAI/bge-m3'), or use a local
directory via config['semantic_model']['name']. Set local_files_only=True
for strictly offline use. There is no silent replacement for BGE-M3.
"""
_SEMANTIC_MODEL = None
_SEMANTIC_MODEL_KEY = None

def _semantic_model_config(model_config: Optional[dict] = None) -> dict:
    if model_config is not None and not isinstance(model_config, dict):
        raise ValueError("semantic_model must be a dict")
    cfg = {**DEFAULT_CONFIG["semantic_model"], **(model_config or {})}
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("semantic_model.name must be a non-empty string")
    batch = cfg["batch_size"]
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("semantic_model.batch_size must be a positive integer")
    if cfg.get("device") is not None and not isinstance(cfg["device"], str):
        raise ValueError("semantic_model.device must be a string or None")
    if "local_files_only" in cfg and not isinstance(cfg["local_files_only"], bool):
        raise ValueError("semantic_model.local_files_only must be a boolean")
    return cfg

_LOCAL_BGE_REVISION = "9a0624b896d81da7492a910ffa53731274b6cf3d"


def _local_semantic_model_directory() -> Optional[str]:
    """Prefer the project model installation without depending on HF cache layout."""
    local = Path(__file__).resolve().parent / ".venv" / "models" / "bge-m3"
    if (local / "config.json").is_file() and (local / "model.safetensors").is_file():
        return str(local)
    return None


def _get_semantic_model(model_config: Optional[dict] = None):
    global _SEMANTIC_MODEL, _SEMANTIC_MODEL_KEY
    cfg = _semantic_model_config(model_config)
    name, device = cfg["name"], cfg.get("device")
    key = (name, device, cfg.get("revision"), cfg.get("cache_folder"), cfg.get("local_files_only", False))
    if _SEMANTIC_MODEL is not None and _SEMANTIC_MODEL_KEY == key:
        return _SEMANTIC_MODEL
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    try:
        from sentence_transformers import SentenceTransformer
        options = {key: cfg[key] for key in ("revision", "cache_folder", "local_files_only") if key in cfg}
        constructor_name = name
        if name == "BAAI/bge-m3" and cfg.get("revision") in (None, _LOCAL_BGE_REVISION):
            local_directory = _local_semantic_model_directory()
            if local_directory is not None:
                constructor_name = local_directory
                options.pop("revision", None)  # A local snapshot already identifies its revision.
        model = SentenceTransformer(constructor_name, device=device, **options)
    except Exception as exc:
        raise SemanticModelUnavailableError(_MODEL_HELP.format(
            name=name, reason=f"{type(exc).__name__}: {exc}")) from exc
    _SEMANTIC_MODEL, _SEMANTIC_MODEL_KEY = model, key
    return model

def semantic_relevance(query: str, passages: list[dict], model_config: Optional[dict] = None) -> np.ndarray:
    texts = [_text_of(passage) for passage in passages]
    scores = np.zeros(len(texts), dtype=float)
    filled = [i for i, text in enumerate(texts) if text]
    if not filled:
        return scores
    cfg = _semantic_model_config(model_config)
    embeddings = np.asarray(_get_semantic_model(cfg).encode(
        [normalize_text(query)] + [normalize_text(texts[i]) for i in filled],
        batch_size=cfg["batch_size"], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(filled) + 1 or not np.isfinite(embeddings).all():
        raise RuntimeError("semantic encoder returned invalid embeddings")
    scores[filled] = embeddings[1:] @ embeddings[0]
    return np.clip(scores, 0, 1)

def score_query_relevance(query: str, passages: list[dict], weights: dict, model_config=None) -> np.ndarray:
    weights = validate_relevance_weights(weights)
    lex = lexical_relevance(query, passages)
    sem = semantic_relevance(query, passages, model_config) if weights["semantic_weight"] else np.zeros_like(lex)
    return np.clip(weights["lexical_weight"] * lex + weights["semantic_weight"] * sem, 0, 1)

def _devanagari_ratio(text: str) -> float:
    return devanagari_ratio(text)

def _keyword_hits(text: str, lexicon: set[str]) -> int:
    for category, aliases in FEATURE_LEXICONS.items():
        if lexicon is aliases or lexicon == aliases:
            return len(matched_concepts(text, category))
    tokens = unicode_tokens(text)
    return sum(any(tokens[i:i + len(needle)] == needle for i in range(len(tokens) - len(needle) + 1))
               for term in lexicon if (needle := unicode_tokens(term)))

def _context_score(features: dict, weights: dict) -> float:
    return _clip01(sum(weights[weight] * features[feature] for feature, weight in CONTEXT_WEIGHT_KEYS.items()))

def compute_indian_context(query: str, passage: dict, weights: dict) -> dict:
    weights = _normalize_weights(weights, DEFAULT_CONFIG["context_weights"], "context_weights")
    features = extract_context_features(query, _text_of(passage))
    return {"score": _context_score(features, weights),
            "breakdown": {key: round(value, 3) for key, value in features.items()}}

def _recency_score(pub_date: Optional[str], reference_date=None) -> float:
    reference = _reference_date(reference_date)
    if not isinstance(pub_date, str) or not pub_date:
        return 0.5
    try:
        published = date.fromisoformat(pub_date)
    except ValueError:
        return 0.5
    if published > reference:
        return 0.5
    years_old = (reference - published).days / 365.25
    return _clip01(1 - min(years_old, 15) / 30)

def _duplication_penalty(idx: int, passages: list[dict], sim_matrix: np.ndarray, threshold=0.92) -> float:
    count = sum(j != idx and sim_matrix[idx, j] >= threshold for j in range(len(passages)))
    return 1.0 / (1 + count)

def compute_source_signal(passage: dict, dup_penalty: float, reference_date=None) -> dict:
    meta, extraction = passage.get("source_metadata") or {}, passage.get("extraction_metadata") or {}
    confidence = extraction.get("ocr_confidence")
    if confidence is not None:
        ocr_score = _finite_number(confidence, "ocr_confidence")
        if not 0 <= ocr_score <= 1:
            raise ValueError("ocr_confidence must be in [0, 1]")
    else:
        # Unknown confidence of scanned text is not equivalent to known clean text.
        method = (extraction.get("method") or "").casefold()
        ocr_score = 0.5 if "ocr" in method else 1.0
    source_type = meta.get("source_type")
    if isinstance(source_type, str):
        source_type = source_type.strip().casefold()
    raw_url = (meta.get("url") or "").strip()
    try:
        parsed = urlparse(raw_url)
        hostname, port = parsed.hostname, parsed.port
        traceable_url = (
            parsed.scheme in ("http", "https") and bool(hostname)
            and not any(character.isspace() for character in raw_url)
            and (port is None or port > 0)
        )
    except ValueError:
        # Broken IPv6 brackets and invalid ports occur in scraped metadata.
        traceable_url = False
    traceable = traceable_url or bool(extraction.get("page_number"))
    features = {
        "source_type_score": SOURCE_TYPE_PRIOR.get(source_type, SOURCE_TYPE_PRIOR[None]),
        "attribution_score": 0.7 if any(
            isinstance(meta.get(field), str) and meta[field].strip()
            for field in ("author", "organization")
        ) else 0.3,
        "date_score": _recency_score(meta.get("publication_date"), reference_date),
        "citation_score": 0.8 if meta.get("has_citations") is True else 0.3,
        "ocr_score": ocr_score, "provenance_score": 1.0 if traceable else 0.4,
        "duplication_score": _clip01(dup_penalty),
    }
    return {"score": _clip01(float(np.mean(list(features.values())))),
            "breakdown": {key: round(value, 3) for key, value in features.items()}}

def normalize_features(raw: dict) -> dict:
    return {key: _clip01(value) for key, value in raw.items()}

def combine_scores(features: dict, ranking_weights: dict) -> float:
    weights = _normalize_weights(ranking_weights, DEFAULT_CONFIG["ranking_weights"], "ranking_weights")
    return _clip01(
        weights["query_relevance_weight"] * features["query_relevance"]
        + weights["indian_context_weight"] * features["indian_context_relevance"]
        + weights["evidence_weight"] * features["source_signal"])

def _extraction_config(cfg: dict) -> dict:
    """Settings that affect cached values, independently of ranking weights."""
    return {
        "schema_version": 2,
        "reference_date": _reference_date(cfg["evidence"]["reference_date"]).isoformat(),
        "duplicate_threshold": cfg["evidence"]["duplicate_threshold"],
        "related_text_threshold": cfg["evidence"]["related_text_threshold"],
        "semantic_model": {
            "name": cfg["semantic_model"]["name"],
            "revision": cfg["semantic_model"].get("revision"),
        },
    }


def prepare_features(query: str, passages: list[dict], config=None) -> list[dict]:
    """Extract label-independent features once; use the same settings when ranking.

    Semantic scores are omitted for explicit lexical-only configurations. Such
    prepared features must not subsequently be reused with positive semantic weight.
    """
    cfg = validated_config(config)
    validate_inputs(query, passages)
    if not passages:
        return []
    # Resolve the implicit date once so every passage and its cache signature
    # use the same day, including a preparation that crosses midnight.
    cfg["evidence"]["reference_date"] = _reference_date(
        cfg["evidence"]["reference_date"]
    ).isoformat()
    extraction_config = _extraction_config(cfg)
    lexical = lexical_relevance(query, passages)
    semantic_enabled = cfg["relevance_weights"]["semantic_weight"] > 0
    semantic = semantic_relevance(query, passages, cfg["semantic_model"]) if semantic_enabled else np.zeros_like(lexical)
    texts = [_text_of(passage) for passage in passages]
    try:
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(
            [normalize_text(text) or " " for text in texts])
        similarities = cosine_similarity(matrix)
    except ValueError:
        similarities = np.eye(len(passages))
    source_support = analyze_source_support(
        passages, similarities, cfg["evidence"]["duplicate_threshold"],
        cfg["evidence"]["related_text_threshold"])
    prepared = []
    for i, passage in enumerate(passages):
        duplicate = _duplication_penalty(i, passages, similarities, cfg["evidence"]["duplicate_threshold"])
        evidence = compute_source_signal(passage, duplicate, cfg["evidence"]["reference_date"])
        prepared.append({
            "passage_id": passage["passage_id"], "text": texts[i],
            "lexical_score": float(lexical[i]), "semantic_score": float(semantic[i]),
            "semantic_available": semantic_enabled,
            "extraction_config": copy.deepcopy(extraction_config),
            "source_support": source_support[i],
            "context_features": extract_context_features(query, texts[i]),
            "source_signal": evidence["score"], "evidence_features": evidence["breakdown"],
            "provenance": {"source_metadata": copy.deepcopy(passage.get("source_metadata") or {}),
                           "extraction_metadata": copy.deepcopy(passage.get("extraction_metadata") or {})},
        })
    return prepared

def rank_prepared(prepared: list[dict], config=None, top_k=None) -> list[dict]:
    """Rank cached features; input order breaks exact ties deterministically.

    Generated features carry an extraction signature. Changing the reference
    date, duplicate threshold, or active semantic model requires preparation
    again. Weight, filter and output changes reuse the extracted values.
    Hand-built legacy feature dictionaries without a signature remain accepted.
    """
    _validate_top_k(top_k)
    cfg = validated_config(config)
    expected_extraction = _extraction_config(cfg)
    results = []
    for feature in prepared:
        if not feature["text"].strip():
            continue
        signature = feature.get("extraction_config")
        if signature is not None:
            fields = ["schema_version", "reference_date", "duplicate_threshold", "related_text_threshold"]
            if cfg["relevance_weights"]["semantic_weight"]:
                fields.append("semantic_model")
            if not isinstance(signature, dict) or any(
                signature.get(field) != expected_extraction[field] for field in fields
            ):
                raise ValueError(
                    "cached feature extraction settings differ; recompute with the requested config"
                )
        if cfg["relevance_weights"]["semantic_weight"] and feature.get("semantic_available") is False:
            raise ValueError("cached features omit semantic scores; recompute with semantic scoring enabled")
        relevance = _clip01(cfg["relevance_weights"]["lexical_weight"] * feature["lexical_score"]
                            + cfg["relevance_weights"]["semantic_weight"] * feature["semantic_score"])
        if cfg["filter"]["enabled"] and relevance < cfg["filter"]["threshold"]:
            continue
        context = _context_score(feature["context_features"], cfg["context_weights"])
        final = combine_scores({"query_relevance": relevance, "indian_context_relevance": context,
                                "source_signal": feature["source_signal"]}, cfg["ranking_weights"])
        item = {"passage_id": feature["passage_id"], "text": feature["text"],
                "query_relevance": round(relevance, 3), "indian_context_relevance": round(context, 3),
                "source_signal": round(feature["source_signal"], 3), "final_score": round(final, 3)}
        if cfg["output"]["include_feature_breakdown"]:
            item["feature_breakdown"] = {
                "indian_context": {key: round(value, 3) for key, value in feature["context_features"].items()},
                "evidence": copy.deepcopy(feature["evidence_features"]),
            }
        if cfg["output"]["include_provenance"]:
            item["provenance"] = copy.deepcopy(feature["provenance"])
        if cfg["output"]["include_source_support"]:
            if "source_support" not in feature:
                raise ValueError("cached features lack source-support indicators; recompute")
            item["source_support"] = copy.deepcopy(feature["source_support"])
        results.append((final, item))
    results.sort(key=lambda pair: pair[0], reverse=True)
    ranked = [{"rank": i, **item} for i, (_, item) in enumerate(results, 1)]
    return ranked if top_k is None else ranked[:top_k]

def rank_evidence(query: str, passages: list[dict], top_k: Optional[int] = None,
                  config: Optional[dict] = None) -> list[dict]:
    _validate_top_k(top_k)
    cfg = validated_config(config)
    validate_inputs(query, passages)
    return rank_prepared(prepare_features(query, passages, cfg), cfg, top_k)

def load_jsonl_dataset(path: str) -> dict[str, dict]:
    grouped = {}
    with open(path, encoding="utf-8") as dataset:
        for number, line in enumerate(dataset, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {number} of {path}: {exc}") from exc
            qid, query = row["query_id"], row["query"]
            if qid in grouped and grouped[qid]["query"] != query:
                raise ValueError(f"inconsistent query text for {qid}")
            grouped.setdefault(qid, {"query": query, "passages": []})["passages"].append({
                "passage_id": row["passage_id"], "text": row.get("text", ""),
                "source_metadata": row.get("source_metadata", {}),
                "extraction_metadata": row.get("extraction_metadata", {}),
            })
    for group in grouped.values():
        validate_inputs(group["query"], group["passages"])
    return grouped

def print_dataset_rankings(dataset_path: str, config=None) -> None:
    grouped = load_jsonl_dataset(dataset_path)
    print(f"Dataset: {dataset_path}\nQueries: {len(grouped)}")
    for query_id, group in grouped.items():
        print(f"\n{query_id}: {group['query']}")
        for row in rank_evidence(group["query"], group["passages"], config=config):
            print(f"{row['rank']:<4}{row['passage_id']:<14} "
                  f"query={row['query_relevance']:.3f} context={row['indian_context_relevance']:.3f} "
                  f"evidence={row['source_signal']:.3f} final={row['final_score']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default=str(Path(__file__).parent / "evaluation/sample_dataset.jsonl"))
    parser.add_argument("--config", help="JSON config exported by optimization")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    print_dataset_rankings(args.dataset, settings)
