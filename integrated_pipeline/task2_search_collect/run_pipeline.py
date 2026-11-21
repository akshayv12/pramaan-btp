"""
Run Task 1 and Task 2. Use --with-task3 to extract the saved documents, or
--with-task4 to extract them and rank the passages with Task 4.

Example:
    python run_pipeline.py --query "What is the Green Revolution?" --max-queries 4
    python run_pipeline.py --with-task4 --query "What is the Green Revolution?"
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


TASK2_DIR = Path(__file__).resolve().parent
TASK1_DIR = TASK2_DIR.parent / "Task1_Multilingual_Query_Generator"

sys.path.insert(0, str(TASK1_DIR))
sys.path.insert(0, str(TASK2_DIR))

from app import generate_queries
from main import OUTPUT_FILE, load_task1_input, merge_duplicates, save_output, search_and_collect


MAX_QUERIES = 4
TOP_K = 10
TASK1_OUTPUT = Path(os.environ.get("TASK1_OUTPUT_PATH", TASK2_DIR / "task1_input.json"))


def run_pipeline(user_queries, max_queries=MAX_QUERIES, top_k=TOP_K):
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    if not user_queries:
        raise ValueError("At least one user query is required.")

    task1_payload = [
        generate_queries(query, max_queries).model_dump(mode="json")
        for query in user_queries
    ]
    if len(task1_payload) == 1:
        task1_payload = task1_payload[0]

    TASK1_OUTPUT.write_text(
        json.dumps(task1_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[TASK 1] Saved handoff: {TASK1_OUTPUT}")

    task1_objects = load_task1_input(TASK1_OUTPUT)
    all_documents = []
    combined_stats = None
    unique_top_level_urls = set()

    for index, task1_object in enumerate(task1_objects, start=1):
        print(
            f"\n[TASK 2] Processing question {index}/{len(task1_objects)}: "
            f"{task1_object.get('original_query')}"
        )
        documents, stats = search_and_collect(
            task1_object,
            top_k=top_k,
            download=True,
            query_index=index,
        )
        all_documents.extend(documents)
        unique_top_level_urls.update(
            stats.get("unique_top_level_url_strings", [])
        )

        if combined_stats is None:
            combined_stats = stats
        else:
            for key in (
                "raw_count",
                "total_documents",
                "ocr_count",
                "direct_count",
                "robots_excluded",
                "meta_robots_excluded",
                "licence_excluded",
                "unsupported_format_excluded",
                "too_small_excluded",
                "download_success",
                "download_failed",
                "linked_resources",
                "licence_uncertain",
                "collection_time_seconds",
            ):
                combined_stats[key] = combined_stats.get(key, 0) + stats.get(key, 0)
            combined_stats["variants_searched"] = (
                combined_stats.get("variants_searched", [])
                + stats.get("variants_searched", [])
            )
            for key in (
                "original_query",
                "normalized_query",
                "query_slug",
                "key_terms",
                "detected_language",
                "task1_schema_version",
            ):
                existing = combined_stats.get(key)
                if not isinstance(existing, list):
                    existing = [existing]
                combined_stats[key] = existing + [stats.get(key)]

    if len(task1_objects) > 1:
        all_documents = merge_duplicates(all_documents)
        combined_stats["unique_top_level_url_strings"] = sorted(unique_top_level_urls)
        combined_stats["unique_top_level_urls"] = len(
            combined_stats["unique_top_level_url_strings"]
        )
        combined_stats["total_documents"] = len(all_documents)

    save_output(all_documents, combined_stats, top_k)
    print("[DONE] Task 3 input: task3_input.json")
    # Task 2 resolves OUTPUT_FILE and its relative download paths in the working directory.
    return TASK1_OUTPUT, Path(OUTPUT_FILE).resolve()


def run_full_pipeline(user_queries, max_queries=MAX_QUERIES, top_k=TOP_K, *, output=None):
    """Run all three stages. Keep the Task 2 handoff available if extraction fails."""
    if output is not None and Path(output).resolve() in {TASK1_OUTPUT.resolve(), Path(OUTPUT_FILE).resolve()}:
        raise ValueError("Task 3 output must differ from the two upstream handoffs")
    task1_path, manifest = run_pipeline(user_queries, max_queries, top_k)
    destination = Path(output).resolve() if output else manifest.with_name("task3_output.json")
    script = TASK2_DIR.parent / "Indian-Context-Retrieval-Summarization-task-3-baseline" / "src" / "process_manifest.py"
    # The two stages both have a module named main. Each keeps its own import context.
    sys.stdout.flush()
    subprocess.run([sys.executable, "-X", "utf8", str(script), str(manifest), "--output", str(destination)], check=True)
    return task1_path, manifest, destination


TASK4_DIR = TASK2_DIR.parent
TASK4_SCRIPT = TASK4_DIR / "process_task3_output.py"
TASK4_TOP_K = 10


def task4_python():
    """Task 4 needs the root environment that holds PyTorch and the local BGE-M3 model."""
    override = os.environ.get("TASK4_PYTHON")
    if override:
        return override
    for candidate in (TASK4_DIR / ".venv" / "Scripts" / "python.exe", TASK4_DIR / ".venv" / "bin" / "python"):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def run_task4_pipeline(user_queries, max_queries=MAX_QUERIES, top_k=TOP_K, *, task3_output=None,
                       output=None, task4_config=None, task4_top_k=TASK4_TOP_K):
    """Run Tasks 1-4. Return the Task 1, Task 2, Task 3 and Task 4 paths.

    A Task 3 batch with individual document failures (exit code 1) still writes its
    output, so Task 4 ranks the passages that were extracted; the caller can inspect
    the Task 3 summary. A fatal Task 3 error (any other nonzero code) stops the run.
    """
    upstream = {TASK1_OUTPUT.resolve(), Path(OUTPUT_FILE).resolve()}
    if task3_output is not None and Path(task3_output).resolve() in upstream:
        raise ValueError("Task 3 output must differ from the two upstream handoffs")
    if output is not None:
        protected = upstream | {Path(task3_output or Path(OUTPUT_FILE).with_name("task3_output.json")).resolve()}
        if Path(output).resolve() in protected:
            raise ValueError("Task 4 output must differ from the three upstream handoffs")
    if isinstance(task4_top_k, bool) or not isinstance(task4_top_k, int) or task4_top_k < 0:
        raise ValueError("task4_top_k must be a non-negative integer")
    task1_path, manifest = run_pipeline(user_queries, max_queries, top_k)
    task3_path = Path(task3_output).resolve() if task3_output else manifest.with_name("task3_output.json")
    task3_script = TASK2_DIR.parent / "Indian-Context-Retrieval-Summarization-task-3-baseline" / "src" / "process_manifest.py"
    sys.stdout.flush()
    task3 = subprocess.run([sys.executable, "-X", "utf8", str(task3_script), str(manifest), "--output", str(task3_path)])
    if task3.returncode not in (0, 1) or not task3_path.is_file():
        raise subprocess.CalledProcessError(task3.returncode, task3.args)
    if task3.returncode == 1:
        print("[TASK 3] Some documents failed; ranking the passages that were extracted.")
    destination = Path(output).resolve() if output else task3_path.with_name("task4_output.json")
    command = [task4_python(), "-X", "utf8", str(TASK4_SCRIPT), str(task3_path), "--output", str(destination),
               "--top-k", str(task4_top_k)]
    if task4_config is not None:
        command += ["--config", str(Path(task4_config).resolve())]
    sys.stdout.flush()
    subprocess.run(command, check=True)
    return task1_path, manifest, task3_path, destination


def run_task5_pipeline(user_queries, max_queries=MAX_QUERIES, top_k=TOP_K, *, task3_output=None,
                       output=None, task4_config=None, task4_top_k=TASK4_TOP_K,
                       task5_output=None, task5_provider="auto", task5_model=None,
                       task5_language=None, task5_strict=False, task5_min_query_rel=0.10):
    """Extend the supplied runner without replacing its Tasks 1–4 implementation."""
    task3_path = Path(task3_output or Path(OUTPUT_FILE).with_name("task3_output.json")).resolve()
    task4_path = Path(output).resolve() if output else task3_path.with_name("task4_output.json")
    destination = Path(task5_output).resolve() if task5_output else task4_path.with_name("task5_output.json")
    protected = {TASK1_OUTPUT.resolve(), Path(OUTPUT_FILE).resolve(), task3_path, task4_path}
    if destination in protected:
        raise ValueError("Task 5 output must differ from all four upstream handoffs")
    if task5_provider not in {"auto", "extractive", "gemini", "ollama"}:
        raise ValueError("Unsupported Task 5 provider")
    if task5_language not in {None, "hi", "en"}:
        raise ValueError("Task 5 language must be hi or en")
    if isinstance(task5_min_query_rel, bool) or not 0 <= task5_min_query_rel <= 1:
        raise ValueError("Task 5 relevance threshold must be in [0, 1]")
    paths = run_task4_pipeline(user_queries, max_queries, top_k, task3_output=task3_output,
                              output=output, task4_config=task4_config, task4_top_k=task4_top_k)
    command = [os.environ.get("TASK5_PYTHON", sys.executable), "-X", "utf8",
               str(TASK4_DIR / "process_task4_output.py"), str(paths[3]),
               "--output", str(destination), "--provider", task5_provider]
    command += ["--min-query-rel", str(task5_min_query_rel)]
    if task5_model:
        command += ["--model", task5_model]
    if task5_language:
        command += ["--language", task5_language]
    if task5_strict:
        command.append("--strict")
    sys.stdout.flush()
    subprocess.run(command, check=True)
    return (*paths, destination)


def main():
    parser = argparse.ArgumentParser(
        description="Generate queries, collect documents, and optionally run Task 3 extraction and Task 4 ranking."
    )
    parser.add_argument(
        "--query",
        nargs="+",
        required=True,
        help="One or more user questions to process in the same batch.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=MAX_QUERIES,
        help="Number of Task 1 search-query variants (2-10).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Maximum search results per Task 1 query.",
    )
    parser.add_argument("--with-task3", action="store_true", help="Extract the collected documents after Task 2")
    parser.add_argument("--task3-output", type=Path, help="Task 3 JSON path for --with-task3 or --with-task4")
    parser.add_argument("--with-task4", action="store_true",
                        help="Also extract (Task 3) and rank the passages (Task 4)")
    parser.add_argument("--with-task5", action="store_true", help="Run all five tasks and produce a cited answer with reliability")
    parser.add_argument("--task5-output", type=Path)
    parser.add_argument("--task5-provider", choices=["auto", "extractive", "gemini", "ollama"], default="auto")
    parser.add_argument("--task5-model")
    parser.add_argument("--task5-language", choices=["hi", "en"])
    parser.add_argument("--task5-strict", action="store_true")
    parser.add_argument("--task5-min-query-rel", type=float, default=0.10)
    parser.add_argument("--task4-output", type=Path, help="Final JSON path for --with-task4")
    parser.add_argument("--task4-config", type=Path, help="Task 4 JSON config (default: configs/deployment.json)")
    parser.add_argument("--task4-top-k", type=int, default=TASK4_TOP_K,
                        help="Ranked passages kept per question by Task 4")
    args = parser.parse_args()
    if args.with_task5:
        args.with_task4 = True
    if (args.task5_output or args.task5_model or args.task5_language or args.task5_strict or args.task5_provider != "auto" or args.task5_min_query_rel != 0.10) and not args.with_task5:
        parser.error("Task 5 settings require --with-task5")
    if args.task3_output and not (args.with_task3 or args.with_task4):
        parser.error("--task3-output requires --with-task3 or --with-task4")
    if (args.task4_output or args.task4_config) and not args.with_task4:
        parser.error("--task4-output and --task4-config require --with-task4")
    if args.with_task4:
        try:
            run = run_task5_pipeline if args.with_task5 else run_task4_pipeline
            task5_options = dict(task5_output=args.task5_output, task5_provider=args.task5_provider,
                                 task5_model=args.task5_model, task5_language=args.task5_language,
                                 task5_strict=args.task5_strict,
                                 task5_min_query_rel=args.task5_min_query_rel) if args.with_task5 else {}
            paths = run(
                args.query, args.max_queries, args.top_k, task3_output=args.task3_output,
                output=args.task4_output, task4_config=args.task4_config, task4_top_k=args.task4_top_k,
                **task5_options)
            task3_path = paths[2]
        except subprocess.CalledProcessError as exc:
            parser.exit(exc.returncode)
        except ValueError as exc:
            parser.error(str(exc))
        summary = json.loads(task3_path.read_text(encoding="utf-8"))["task3_output"]["summary"]
        # Match Task 3's convention: completed output with document failures exits 1.
        if summary.get("failed"):
            parser.exit(1)
    elif args.with_task3:
        try:
            run_full_pipeline(args.query, args.max_queries, args.top_k, output=args.task3_output)
        except subprocess.CalledProcessError as exc:
            parser.exit(exc.returncode)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        run_pipeline(args.query, args.max_queries, args.top_k)


if __name__ == "__main__":
    main()
