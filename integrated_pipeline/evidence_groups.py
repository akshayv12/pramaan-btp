"""Conservative source identity, duplicate and reviewed-support indicators.

Text similarity finds candidates only. Neither a different hostname nor matching
wording establishes factual agreement or independent evidence. These indicators
do not automatically increase the ranking or source-quality score.
"""
from __future__ import annotations
import hashlib
import math
from collections import defaultdict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from context_features import normalize_text

TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}

def _identifier(value):
    return value.strip() if isinstance(value, str) and value.strip() else None

def canonical_source_url(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
        host = parsed.hostname
        port = parsed.port
        if parsed.scheme.lower() not in ("http", "https") or not host:
            return None
        if any(char.isspace() for char in value) or parsed.username or parsed.password:
            return None
        host = host.casefold()
        netloc = "[" + host + "]" if ":" in host else host
        if port is not None and not ((parsed.scheme == "https" and port == 443) or
                                     (parsed.scheme == "http" and port == 80)):
            if not 1 <= port <= 65535:
                return None
            netloc += ":" + str(port)
        query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                 if not key.casefold().startswith("utm_") and key.casefold() not in TRACKING_PARAMETERS]
        # Preserve meaningful path/query differences; strip fragments only.
        return urlunsplit((parsed.scheme.casefold(), netloc, parsed.path or "/", urlencode(query), ""))
    except (ValueError, UnicodeError):
        return None

def _identity(passage):
    meta = passage.get("source_metadata") or {}
    extraction = passage.get("extraction_metadata") or {}
    document_id = (_identifier(meta.get("document_id")) or _identifier(passage.get("document_id"))
                   or _identifier(extraction.get("document_id")))
    url = canonical_source_url(meta.get("url"))
    if document_id:
        document = "document:" + document_id
        identity_basis = "declared_document_id"
    elif url:
        document = "url:" + url
        identity_basis = "canonical_url"
    else:
        document = "unidentified:" + passage["passage_id"]
        identity_basis = "passage_fallback"
    declared_group = _identifier(meta.get("source_group_id"))
    organization = _identifier(meta.get("organization"))
    if declared_group:
        source_group, group_basis = "declared:" + declared_group, "declared_source_group"
    elif organization:
        source_group, group_basis = "organization:" + normalize_text(organization), "organization_metadata"
    elif url:
        hostname = urlsplit(url).hostname
        source_group, group_basis = "hostname:" + hostname.removeprefix("www."), "hostname_proxy"
    else:
        source_group, group_basis = None, "unknown"
    origin = _identifier(meta.get("origin_document_id")) or _identifier(meta.get("copied_from_document_id"))
    return {
        "document_id": document, "document_identity_basis": identity_basis,
        "source_group_id": source_group, "source_group_basis": group_basis,
        "declared_origin": "document:" + origin if origin else document,
        "canonical_url": url,
    }

def validate_support_metadata(passage):
    meta = passage.get("source_metadata") or {}
    for field in ("document_id", "source_id", "source_group_id", "origin_document_id",
                  "copied_from_document_id"):
        if meta.get(field) is not None and not _identifier(meta[field]):
            raise ValueError(f"source_metadata.{field} must be a nonempty string or None")
    origin = _identifier(meta.get("origin_document_id"))
    copied = _identifier(meta.get("copied_from_document_id"))
    if origin and copied and origin != copied:
        raise ValueError("origin_document_id and copied_from_document_id conflict")
    records = meta.get("support_annotations", [])
    if records is None:
        return
    if not isinstance(records, list):
        raise ValueError("support_annotations must be a list")
    text = normalize_text((passage.get("text") or ""))
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each support annotation must be an object")
        if record.get("review_status") == "pending":
            continue
        if record.get("review_status") != "reviewed":
            raise ValueError("support annotation review_status must be pending or reviewed")
        if not _identifier(record.get("claim_id")) or not _identifier(record.get("reviewer_id")):
            raise ValueError("reviewed support requires claim_id and declared reviewer_id")
        if record.get("stance") not in ("supports", "contradicts", "mentions"):
            raise ValueError("support stance must be supports, contradicts, or mentions")
        span = _identifier(record.get("supporting_span"))
        if not span or normalize_text(span) not in text:
            raise ValueError("reviewed supporting_span must occur in the passage text")

def analyze_source_support(passages, similarities, duplicate_threshold=0.92, related_threshold=0.65):
    """Return per-passage metadata indicators over ALL input candidates.

    Reviewed stances are supplied declarations. The code records them and checks
    their spans; it does not establish reviewer identity, truth or independence.
    """
    count = len(passages)
    if len(similarities) != count or any(len(row) != count for row in similarities):
        raise ValueError("similarity matrix must match the passage collection")
    if not 0 <= related_threshold <= duplicate_threshold <= 1:
        raise ValueError("related threshold must be <= duplicate threshold, both in [0,1]")
    identities = [_identity(passage) for passage in passages]
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)
    for identity in identities:
        union(identity["document_id"], identity["declared_origin"])
    origins = [find(identity["document_id"]) for identity in identities]
    origin_members = defaultdict(list)
    for index, origin in enumerate(origins):
        origin_members[origin].append(index)
    origin_source_groups = {}
    for origin, indexes in origin_members.items():
        originals = [i for i in indexes if identities[i]["document_id"] == identities[i]["declared_origin"]]
        candidates = originals or indexes
        groups = {identities[i]["source_group_id"] for i in candidates if identities[i]["source_group_id"] is not None}
        origin_source_groups[origin] = next(iter(groups)) if len(groups) == 1 else None
    claims = defaultdict(lambda: {"supports": [], "contradicts": [], "mentions": []})
    passage_claims = []
    for index, passage in enumerate(passages):
        validate_support_metadata(passage)
        invoked = set()
        for record in (passage.get("source_metadata") or {}).get("support_annotations") or []:
            if record.get("review_status") != "reviewed":
                continue
            claim = record["claim_id"].strip()
            invoked.add(claim)
            claims[claim][record["stance"]].append(index)
        passage_claims.append(invoked)
    results = []
    for index, identity in enumerate(identities):
        related, duplicates = [], []
        for other in range(count):
            if other == index or not (passages[other].get("text") or "").strip():
                continue
            score = float(similarities[index][other])
            if not math.isfinite(score):
                raise ValueError("similarities must be finite")
            if score >= related_threshold:
                related.append(other)
            if score >= duplicate_threshold:
                duplicates.append(other)
        same_origin = [other for other in range(count) if other != index and origins[other] == origins[index]]
        related_groups = {identities[other]["source_group_id"] for other in [index, *related]
                          if identities[other]["source_group_id"] is not None}
        summaries = []
        for claim in sorted(passage_claims[index]):
            record = {"claim_id": claim, "provenance": "supplied_review_declarations"}
            for stance, prefix in (("supports", "supporting"), ("contradicts", "contradicting"),
                                   ("mentions", "mentioning")):
                indexes = claims[claim][stance]
                record[prefix + "_passage_ids"] = sorted({passages[i]["passage_id"] for i in indexes})
                record[prefix + "_document_origins"] = len({origins[i] for i in indexes})
                record[prefix + "_source_groups"] = len({
                    origin_source_groups[origins[i]] for i in indexes
                    if origin_source_groups[origins[i]] is not None})
            record["has_reviewed_disagreement"] = bool(
                record["supporting_passage_ids"] and record["contradicting_passage_ids"])
            summaries.append(record)
        results.append({
            **{key: value for key, value in identity.items() if key != "declared_origin"},
            "origin_group_id": origins[index],
            "same_origin_passage_ids": sorted(passages[i]["passage_id"] for i in same_origin),
            "near_duplicate_passage_ids": sorted(passages[i]["passage_id"] for i in duplicates),
            "similar_text_passage_ids": sorted(passages[i]["passage_id"] for i in related),
            "similar_text_source_group_count": len(related_groups),
            "similar_text_document_origin_count": len({origins[i] for i in [index, *related]}),
            "source_independence_established": False,
            "text_similarity_establishes_agreement": False,
            "reviewed_claims": summaries,
            "scope": "all_input_candidates_before_filtering",
        })
    return results
