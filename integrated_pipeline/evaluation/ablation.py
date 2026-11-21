"""Compare six weight ablations using the same cached candidate features."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task4_rank_evidence import validated_config
from evaluation.evaluate import (
    DEFAULT_KS, load_dataset, load_feature_cache, prepare_dataset_features, evaluate_ranker,
)


def _renormalize(weights: dict, zero_key: str) -> dict:
    result = dict(weights)
    result[zero_key] = 0.0
    total = sum(result.values())
    if total <= 0:
        raise ValueError("ablation leaves no positive weight")
    return {key: value / total for key, value in result.items()}


def build_variants(config=None) -> dict[str, dict]:
    """Use the supplied trained config as A0; do not tune any ablation."""
    base = validated_config(config)
    variants = {"A0_full_model": copy.deepcopy(base)}
    a1 = copy.deepcopy(base)
    a1["relevance_weights"] = {"lexical_weight": 1.0, "semantic_weight": 0.0}
    variants["A1_no_semantic"] = a1
    a2 = copy.deepcopy(base)
    a2["ranking_weights"] = _renormalize(base["ranking_weights"], "indian_context_weight")
    variants["A2_no_indian_context"] = a2
    a3 = copy.deepcopy(base)
    a3["ranking_weights"] = _renormalize(base["ranking_weights"], "evidence_weight")
    variants["A3_no_evidence"] = a3
    a4 = copy.deepcopy(base)
    a4["context_weights"] = _renormalize(
        _renormalize(base["context_weights"], "w_script"), "w_query_ctx")
    variants["A4_context_as_keywords"] = a4
    a5 = copy.deepcopy(base)
    a5["ranking_weights"] = {
        "query_relevance_weight": 0.34, "indian_context_weight": 0.33,
        "evidence_weight": 0.33,
    }
    variants["A5_alt_weights_34_33_33"] = a5
    return variants


def run_ablation(dataset_path=None, ks=DEFAULT_KS, config=None,
                 prepared_features=None, grouped_dataset=None) -> list[dict]:
    grouped = load_dataset(dataset_path) if grouped_dataset is None else grouped_dataset
    baseline = validated_config(config)
    cache = (prepare_dataset_features(grouped, baseline)
             if prepared_features is None else prepared_features)
    rows = []
    for name, variant in build_variants(baseline).items():
        report = evaluate_ranker(grouped, config=variant, ks=ks, prepared_features=cache)
        row = {"variant": name}
        for k in ks:
            for metric in ("P", "R", "nDCG"):
                row[f"{metric}@{k}"] = report[f"mean_{metric}@{k}"]
        row["context_agreement"] = report["mean_context_agreement_spearman"]
        row["report"] = report
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    for row in rows:
        print(row["variant"], " ".join(
            f"{key}={value:.6f}" for key, value in row.items()
            if isinstance(value, (int, float))))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default=str(_ROOT / "evaluation/sample_dataset.jsonl"))
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--features")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    cache = load_feature_cache(args.features, args.dataset) if args.features else None
    rows = run_ablation(args.dataset, config=config, prepared_features=cache)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
    print_table(rows)


if __name__ == "__main__":
    main()
