"""Run a seeded, train-only twelve-weight experiment and save its evidence.

The query holdout is excluded from optimization. The small supplied benchmark
has informed implementation decisions, so it is not an untouched external test.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from task4_rank_evidence import validated_config
from evaluation.evaluate import (
    DEFAULT_KS, load_dataset, load_feature_cache, prepare_dataset_features, evaluate_ranker, feature_cache_sources,
)
from evaluation.ablation import run_ablation
from evaluation.baselines import run_baselines
from evaluation.calibrate_filter import calibrate_filter as select_filter

WEIGHT_GROUPS = {
    "relevance_weights": ("lexical_weight", "semantic_weight"),
    "context_weights": (
        "w_script", "w_entity", "w_place", "w_institution",
        "w_primary", "w_vocab", "w_query_ctx",
    ),
    "ranking_weights": (
        "query_relevance_weight", "indian_context_weight", "evidence_weight",
    ),
}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def create_split(grouped: dict, seed: int = 42, train_fraction: float = 0.8) -> dict:
    """Freeze by sorted family IDs, ensuring equivalents stay together.

    Optional JSONL query_family values must identify translations/paraphrases
    belonging to the same underlying information need before creating a split.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one")
    families = {}
    for qid, group in sorted(grouped.items()):
        families.setdefault(group.get("query_family", qid), []).append(qid)
    ids = sorted(families)
    if len(ids) < 2:
        raise ValueError("at least two independent query families are required")
    random.Random(seed).shuffle(ids)
    count = max(1, min(len(ids) - 1, int(len(ids) * train_fraction)))
    train_families = set(ids[:count])
    train = sorted(qid for family in train_families for qid in families[family])
    test = sorted(set(grouped) - set(train))
    return {
        "schema_version": 1, "seed": seed, "train_fraction": train_fraction,
        "split_unit": "query_family", "train_query_ids": train, "test_query_ids": test,
        "query_families": {qid: group.get("query_family", qid)
                           for qid, group in sorted(grouped.items())},
    }


def validate_split(split: dict, grouped: dict, dataset_hash: str | None = None) -> None:
    train, test = split["train_query_ids"], split["test_query_ids"]
    if not train or not test or len(train) != len(set(train)) or len(test) != len(set(test)):
        raise ValueError("split requires nonempty, duplicate-free train and test sets")
    if set(train) & set(test) or set(train) | set(test) != set(grouped):
        raise ValueError("split must partition all query IDs without overlap")
    if dataset_hash is not None and split.get("dataset_sha256") != dataset_hash:
        raise ValueError("split dataset hash differs from the supplied dataset")
    families = {qid: group.get("query_family", qid) for qid, group in grouped.items()}
    if split.get("query_families") != families:
        raise ValueError("split family assignments differ from the dataset")
    if {families[qid] for qid in train} & {families[qid] for qid in test}:
        raise ValueError("equivalent query families cross the train/test boundary")


def config_from_params(params: dict, base_config: dict) -> dict:
    config = copy.deepcopy(base_config)
    for section, keys in WEIGHT_GROUPS.items():
        values = {key: float(params[f"{section}.{key}"]) for key in keys}
        total = sum(values.values())
        if total <= 0:
            raise ValueError("sampled weight group must have a positive total")
        config[section] = {key: value / total for key, value in values.items()}
    return validated_config(config)


def make_objective(train_dataset: dict, prepared_features: dict, base_config: dict):
    """Close over training queries only; held-out labels never enter this API."""
    train_cache = {qid: prepared_features[qid] for qid in train_dataset}

    def objective(trial):
        params = {
            f"{section}.{key}": trial.suggest_float(f"{section}.{key}", 0.01, 1.0)
            for section, keys in WEIGHT_GROUPS.items() for key in keys
        }
        config = config_from_params(params, base_config)
        report = evaluate_ranker(train_dataset, config=config, ks=(10,),
                                 prepared_features=train_cache)
        return report["mean_nDCG@10"]
    return objective


def environment_manifest() -> dict:
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "sentence-transformers",
                    "torch", "optuna", "transformers", "tokenizers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": versions,
    }


def run_experiment(dataset_path, output_dir, config=None, trials=300, seed=42,
                   reference_date=None, split_path=None, feature_cache_path=None,
                   calibrate_filter=False, min_retention=0.95) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = reference_date or (config or {}).get("evidence", {}).get("reference_date") or date.today().isoformat()
    date.fromisoformat(reference)
    base = copy.deepcopy(config or {})
    base.setdefault("evidence", {})["reference_date"] = reference
    base = validated_config(base)
    # Ranking weights are selected before the gate; no threshold is selected on test.
    base["filter"] = {"enabled": False, "threshold": 0.0}
    if base["relevance_weights"]["semantic_weight"] <= 0:
        raise ValueError("the twelve-weight experiment requires semantic features")
    grouped = load_dataset(dataset_path)
    digest = sha256_file(dataset_path)
    if split_path:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    else:
        split = create_split(grouped, seed=seed)
        split["dataset_sha256"] = digest
    validate_split(split, grouped, digest)
    write_json(output / "split.json", split)
    write_json(output / "base_config.json", base)
    manifest = {
        "schema_version": 1, "status": "preparing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(dataset_path).resolve()), "dataset_sha256": digest,
        "query_count": len(grouped),
        "passage_count": sum(len(group["passages"]) for group in grouped.values()),
        "split": split, "seed": seed, "requested_trials": trials,
        "sampler": "TPESampler(seed=seed)", "n_jobs": 1,
        "objective": "unrounded mean nDCG@10 on train_query_ids only",
        "search_space": {f"{section}.{key}": {"low": 0.01, "high": 1.0}
                         for section, keys in WEIGHT_GROUPS.items() for key in keys},
        "normalization": "each of the relevance/context/ranking weight groups separately sums to one",
        "reference_date": reference, "metrics_k": list(DEFAULT_KS),
        "base_config": base, "environment": environment_manifest(),
        "source_sha256": {
            str(path.relative_to(_ROOT)): sha256_file(path)
            for path in feature_cache_sources()
        },
        "limitations": [
            "The supplied small development benchmark informed implementation; its query holdout is excluded from optimization but is not an untouched external test.",
            "Annotation authorship and factual/source reliability are not independently established.",
            "No synthetic extensions are included in the primary experiment.",
            "Recall@10 is saturated on candidate groups with fewer than ten passages; inspect Recall@1/3 and nDCG.",
            "Precision uses the available returned count up to K, not a fixed K denominator.",
            "Ablations reuse trained weights without retuning; test results are descriptive and do not select weights.",
            "Ranking optimization is unfiltered. Optional gate calibration uses training labels only; held-out results never select a gate.",
            "The six-query development holdout has been inspected previously; a separately collected, frozen real-source test set is still required.",
        ],
    }
    manifest["development_status"] = "previously inspected development benchmark; not a fresh final test"
    manifest["calibrate_filter"] = bool(calibrate_filter)
    manifest["calibration_policy"] = {
        "objective": "maximize training micro passage F1",
        "minimum_positive_retention": min_retention,
        "positive_label_threshold": 2,
        "tie_break": ["higher positive retention", "lower threshold"],
        "selection_population": "train_query_ids only",
    } if calibrate_filter else None
    # Read an imported cache before writing the new manifest, including when
    # callers choose the original cache directory as their output directory.
    cache = load_feature_cache(feature_cache_path, dataset_path) if feature_cache_path else None
    write_json(output / "manifest.json", manifest)
    if cache is None:
        cache = prepare_dataset_features(grouped, base)
    # Validate reused extraction signatures and IDs without accessing relevance labels.
    for qid, group in grouped.items():
        if qid not in cache or [feature["passage_id"] for feature in cache[qid]] != [
                passage["passage_id"] for passage in group["passages"]]:
            raise ValueError(f"feature cache IDs differ for {qid}")
        from task4_rank_evidence import rank_prepared
        rank_prepared(cache[qid], config=base)
    write_json(output / "features.json", cache)
    train = {qid: grouped[qid] for qid in split["train_query_ids"]}
    test = {qid: grouped[qid] for qid in split["test_query_ids"]}
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    manifest["status"] = "optimizing"
    write_json(output / "manifest.json", manifest)
    study.optimize(make_objective(train, cache, base), n_trials=trials, n_jobs=1)
    chosen = config_from_params(study.best_trial.params, base)
    write_json(output / "chosen_config.json", chosen)
    with (output / "trials.jsonl").open("w", encoding="utf-8") as stream:
        for trial in study.trials:
            stream.write(json.dumps({
                "number": trial.number, "value": trial.value,
                "state": trial.state.name, "params": trial.params,
            }, allow_nan=False) + "\n")
    reports = {}
    for name, cfg in (("baseline", base), ("optimized", chosen)):
        reports[name] = {
            subset: evaluate_ranker(data, config=cfg, prepared_features=cache)
            for subset, data in (("train", train), ("test", test), ("full", grouped))
        }
    ablations = {
        subset: run_ablation(config=chosen, grouped_dataset=data, prepared_features=cache)
        for subset, data in (("train", train), ("test", test), ("full", grouped))
    }
    reports["ablations"] = ablations
    reports["baselines"] = {
        subset: run_baselines(data, config=chosen, prepared_features=cache)
        for subset, data in (("train", train), ("test", test), ("full", grouped))
    }
    write_json(output / "baselines.json", reports["baselines"])
    extra_artifacts = ["baselines.json"]
    deployment_config = None
    if calibrate_filter:
        # Pass an explicitly restricted cache as well as restricted labels.
        train_cache = {qid: cache[qid] for qid in train}
        calibration = {
            "baseline": select_filter(train, train_cache, base, min_retention=min_retention),
            "optimized": select_filter(train, train_cache, chosen, min_retention=min_retention),
            "deployment_profile": "training_selected_optimized",
            "profile_selection": "optimized weights selected by training nDCG@10; threshold selected by training constrained F1; no test metrics used",
            "dataset_sha256": digest, "source_sha256": manifest["source_sha256"],
            "reference_date": reference,
            "test_query_ids_excluded_from_selection": list(split["test_query_ids"]),
            "development_status": manifest["development_status"],
        }
        deployment_config = calibration["optimized"]["deployment_config"]
        baseline_deployment = calibration["baseline"]["deployment_config"]
        write_json(output / "filter_calibration.json", calibration)
        write_json(output / "deployment_config.json", deployment_config)
        write_json(output / "baseline_deployment_config.json", baseline_deployment)
        for name, profile in (("deployment", deployment_config),
                              ("baseline_deployment", baseline_deployment)):
            reports[name] = {
                subset: evaluate_ranker(data, config=profile, prepared_features=cache)
                for subset, data in (("train", train), ("test", test), ("full", grouped))
            }
        manifest["deployment_profile"] = calibration["deployment_profile"]
        manifest["deployment_config"] = deployment_config
        extra_artifacts.extend(["filter_calibration.json", "deployment_config.json",
                                "baseline_deployment_config.json"])
    write_json(output / "reports.json", reports)
    write_json(output / "ablations.json", ablations)
    manifest.update({
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_trials": len(study.trials), "best_trial_number": study.best_trial.number,
        "best_train_objective": study.best_value, "chosen_config": chosen,
        "feature_cache_sha256": sha256_file(output / "features.json"),
        "artifact_sha256": {
            name: sha256_file(output / name) for name in (
                "split.json", "base_config.json", "chosen_config.json",
                "features.json", "trials.jsonl", "reports.json", "ablations.json", *extra_artifacts)
        },
    })
    write_json(output / "manifest.json", manifest)
    print(f"Completed {len(study.trials)} trials; train={len(train)}, test={len(test)}")
    print(f"Best train nDCG@10: {study.best_value:.6f}")
    print(f"Test nDCG@10: {reports['optimized']['test']['mean_nDCG@10']:.6f}")
    print(f"Artifacts: {output.resolve()}")
    return {"manifest": manifest, "reports": reports, "config": chosen,
            "deployment_config": deployment_config}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(_ROOT / "evaluation/sample_dataset.jsonl"))
    parser.add_argument("--config")
    parser.add_argument("--output", default=str(_ROOT / "artifacts/experiment"))
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibrate-filter", action="store_true", help="Calibrate a relevance gate using training labels only")
    parser.add_argument("--min-retention", type=float, default=0.95)
    parser.add_argument("--reference-date")
    parser.add_argument("--split", help="Reuse split.json; requires the same dataset SHA-256")
    parser.add_argument("--features", help="Reuse features.json with identical extraction settings")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    run_experiment(args.dataset, args.output, config=config, trials=args.trials,
                   seed=args.seed, reference_date=args.reference_date,
                   split_path=args.split, feature_cache_path=args.features,
                   calibrate_filter=args.calibrate_filter, min_retention=args.min_retention)


if __name__ == "__main__":
    main()
