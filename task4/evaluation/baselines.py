"""Pure query-relevance baselines, separate from component ablations."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task4_rank_evidence import validated_config
from evaluation.evaluate import DEFAULT_KS, evaluate_ranker, load_dataset, load_feature_cache, prepare_dataset_features


def build_baselines(config=None):
    base = validated_config(config)
    # Query-only comparisons use no gate: one common threshold would not be
    # calibrated equally across lexical, semantic and hybrid score scales.
    base["filter"] = {"enabled": False, "threshold": 0.0}
    variants = {}
    for name, relevance in (
        ("B0_tfidf_query_only", {"lexical_weight": 1.0, "semantic_weight": 0.0}),
        ("B1_semantic_query_only", {"lexical_weight": 0.0, "semantic_weight": 1.0}),
        ("B2_hybrid_query_only", base["relevance_weights"]),
    ):
        cfg = copy.deepcopy(base)
        cfg["relevance_weights"] = dict(relevance)
        cfg["ranking_weights"] = {
            "query_relevance_weight": 1.0, "indian_context_weight": 0.0,
            "evidence_weight": 0.0,
        }
        variants[name] = cfg
    variants["B3_full_model"] = copy.deepcopy(base)
    return variants


def run_baselines(grouped_dataset, config=None, prepared_features=None, ks=DEFAULT_KS):
    cfg = validated_config(config)
    cache = prepare_dataset_features(grouped_dataset, cfg) if prepared_features is None else prepared_features
    # The development JSONL often lists positive examples first. Stable core
    # tie-breaking would otherwise reward that incidental order, especially for
    # cross-script TF-IDF zero ties. Sort inputs without consulting any labels.
    def neutral_key(qid, passage_id):
        framed = json.dumps([qid, passage_id], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(framed.encode("utf-8")).digest()

    neutral_data = {
        qid: {**data, "passages": sorted(
            data["passages"], key=lambda passage: neutral_key(qid, passage["passage_id"]))}
        for qid, data in grouped_dataset.items()
    }
    neutral_cache = {
        qid: sorted(cache[qid], key=lambda feature: neutral_key(qid, feature["passage_id"]))
        for qid in grouped_dataset
    }
    reports = {
        name: evaluate_ranker(neutral_data, config=variant, ks=ks, prepared_features=neutral_cache)
        for name, variant in build_baselines(cfg).items()
    }
    for report in reports.values():
        report["tie_policy"] = (
            "label-independent candidate order by SHA-256 of compact UTF-8 JSON "
            "[query_id,passage_id]; stable core ties preserve this order")
        report["source_order_note"] = (
            "source JSONL is unchanged; the same neutral candidate order is used "
            "for all four baseline comparisons")
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default=str(_ROOT / "evaluation/sample_dataset.jsonl"))
    parser.add_argument("--config")
    parser.add_argument("--features")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    cache = load_feature_cache(args.features, args.dataset) if args.features else None
    result = run_baselines(load_dataset(args.dataset), config, cache)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, report in result.items():
        print(name, f"nDCG@10={report['mean_nDCG@10']:.6f}")


if __name__ == "__main__":
    main()
