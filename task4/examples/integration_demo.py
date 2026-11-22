"""Local Task 4 interface demonstration using constructed inputs only.

The Task 3 and Task 5 formats are proposals, not live teammate integrations.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Task 4 JSON config (default: deployment.json, else cpu.json)")
    parser.add_argument("--fixture", default=str(ROOT / "examples/fixtures/task3_example.json"))
    parser.add_argument("--output", help="Write JSON report to this path")
    parser.add_argument("--query", help="Override the fixture query; relevance is measured, not forced")
    parser.add_argument("--lexical-only", action="store_true", help="Use TF-IDF only; no encoder needed")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from contracts import normalize_task3_package, build_task5_package, task5_consumer_stub
    from task4_rank_evidence import rank_evidence

    config_path = Path(args.config) if args.config else ROOT / "configs/deployment.json"
    if not config_path.exists() and not args.config:
        config_path = ROOT / "configs/cpu.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config = copy.deepcopy(config)
    if args.offline:
        config.setdefault("semantic_model", {})["local_files_only"] = True
    if args.lexical_only:
        config["relevance_weights"] = {"lexical_weight": 1.0, "semantic_weight": 0.0}
    # A positive relevance gate is essential for this local selection example.
    # Keep a deployment threshold when present; otherwise use a documented 0.10.
    gate = config.setdefault("filter", {})
    if not gate.get("enabled") or not isinstance(gate.get("threshold"), (int, float)) or gate["threshold"] <= 0:
        gate.update(enabled=True, threshold=0.10)
    config.setdefault("output", {}).update(include_provenance=True, include_source_support=True)
    raw = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    if args.query is not None:
        raw["query"] = args.query
    normalized = normalize_task3_package(raw)
    ranked = rank_evidence(normalized["query"], normalized["passages"], config=config)
    envelope = build_task5_package(normalized["query"], ranked, provenance=normalized["provenance"])
    consumer_result = task5_consumer_stub(envelope)
    report = {
        "interface_status": "proposed_local_contract_only",
        "live_handoff_available": False,
        "is_simulated": normalized["provenance"]["kind"] == "constructed_example",
        "fixture_provenance": normalized["provenance"],
        "fixture": str(Path(args.fixture).resolve()),
        "config_path": str(config_path.resolve()),
        "config_used": config,
        "candidate_count": len(normalized["passages"]),
        "ranked_count": len(ranked),
        "status": envelope["status"],
        "task5_envelope": envelope,
        "validation": {"input": True, "output": True, "consumer_stub": consumer_result},
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "is_simulated": report["is_simulated"],
                          "candidate_count": report["candidate_count"],
                          "ranked_count": report["ranked_count"], "status": report["status"]}))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
