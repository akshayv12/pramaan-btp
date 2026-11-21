"""Tasks 1–5 entry point, with explicit resume modes for supplied handoff files."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the supplied Tasks 1–4 pipeline plus the actual Task 5 consumer")
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--query", nargs="+", help="Run live collection for one or more questions")
    entry.add_argument("--task3-input", type=Path, help="Resume from Task 3 extraction output")
    entry.add_argument("--task4-input", type=Path, help="Resume directly from ranked Task 4 output")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "latest")
    parser.add_argument("--max-queries", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--task4-top-k", type=int, default=10)
    parser.add_argument("--task4-config", type=Path)
    parser.add_argument("--lexical-only", action="store_true", help="Explicit Task 4 baseline; no BGE-M3 model is loaded")
    parser.add_argument("--provider", choices=["auto", "extractive", "gemini", "ollama"])
    parser.add_argument("--model")
    parser.add_argument("--language", choices=["hi", "en"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min-query-rel", type=float, default=0.10)
    args = parser.parse_args(argv)
    if not 2 <= args.max_queries <= 10 or args.top_k < 1 or args.task4_top_k < 0:
        parser.error("max-queries must be 2–10; top-k positive; task4-top-k nonnegative")
    if args.task4_input and (args.lexical_only or args.task4_config):
        parser.error("Task 4 settings cannot be used when resuming after Task 4")
    from task5.config import Task5Config, load_env_file
    from task5.handoff import process_task4_output, write_json_atomic
    load_env_file(ROOT / ".env")
    load_env_file()
    try:
        config = Task5Config.from_env(provider=args.provider, model=args.model, allow_fallback=not args.strict,
                                     min_query_rel=args.min_query_rel)
    except ValueError as exc:
        parser.error(str(exc))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    final = out / "task5_output.json"
    task4_file = out / "task4_output.json"
    if args.task3_input and args.task3_input.resolve() in {final, task4_file}:
        parser.error("Run output paths must differ from the Task 3 input")
    try:
        if args.task4_input:
            mode = "resume_task4"
            task4_file = args.task4_input.resolve()
        elif args.task3_input:
            mode = "resume_task3"
            from process_task3_output import load_config, process_task3_output
            rank_config = load_config(args.task4_config)
            if args.lexical_only:
                rank_config["relevance_weights"] = {"lexical_weight": 1.0, "semantic_weight": 0.0}
            process_task3_output(args.task3_input, task4_file, config=rank_config, top_k=args.task4_top_k)
        else:
            mode = "live_tasks1_to_5"
            pending_final = out / (".task5_output.pending-" + uuid.uuid4().hex + ".json")
            # Import contexts of upstream modules named app/main stay in their own process.
            command = [sys.executable, str(ROOT / "task2_search_collect" / "run_pipeline.py"),
                       "--with-task5", "--query", *args.query, "--max-queries", str(args.max_queries),
                       "--top-k", str(args.top_k), "--task4-top-k", str(args.task4_top_k),
                       "--task4-output", str(task4_file), "--task5-output", str(pending_final),
                       "--task5-provider", config.provider]
            command += ["--task5-min-query-rel", str(config.min_query_rel)]
            if args.lexical_only:
                if args.task4_config:
                    from process_task3_output import load_config
                    rank_config = load_config(args.task4_config)
                    rank_config["relevance_weights"] = {"lexical_weight": 1.0, "semantic_weight": 0.0}
                    rank_path = out / "task4_config_used.json"
                    write_json_atomic(rank_path, rank_config)
                else:
                    rank_path = ROOT / "configs" / "integration_lexical.json"
                command += ["--task4-config", str(rank_path)]
            elif args.task4_config:
                command += ["--task4-config", str(args.task4_config.resolve())]
            if config.model:
                command += ["--task5-model", config.model]
            if args.language:
                command += ["--task5-language", args.language]
            if args.strict:
                command.append("--task5-strict")
            # All upstream relative downloads/handoffs are contained in this run directory.
            environment = dict(os.environ, TASK1_OUTPUT_PATH=str(out / "task1_input.json"))
            process = subprocess.run(command, cwd=out, env=environment)
            if process.returncode not in (0, 1) or not pending_final.is_file():
                pending_final.unlink(missing_ok=True)
                return process.returncode or 2
            result = json.loads(pending_final.read_text(encoding="utf-8"))
            result["task5_output"]["execution_mode"] = mode
            result["task5_output"]["upstream_exit_code"] = process.returncode
            write_json_atomic(final, result)
            pending_final.unlink(missing_ok=True)
            print(f"Final answers: {final}")
            return process.returncode
        result = process_task4_output(task4_file, final, config=config, target_language=args.language)
        result["task5_output"]["execution_mode"] = mode
        result["task5_output"]["input_file"] = str((args.task3_input or args.task4_input).resolve())
        write_json_atomic(final, result)
        for item in result["task5_output"]["queries"]:
            answer = item["result"]
            print(f"\nQuery: {item['query']}\n{answer['answer']}\nReliability: {answer['reliability']}")
        print(f"\nFinal answers: {final}")
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"[INTEGRATION] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
