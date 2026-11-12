"""End-to-end QA and full-corpus Codec reconstruction benchmarks."""

import argparse
import asyncio
import signal
from pathlib import Path

from .storage import dumps, read_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    suites = p.add_subparsers(dest="suite", required=True)
    for suite in ("e2e", "codec"):
        phases = suites.add_parser(suite).add_subparsers(dest="phase", required=True)
        run = phases.add_parser("run", help="Generate and freeze outputs")
        run.add_argument("--config", type=Path, required=True)
        run.add_argument("--output-dir", type=Path, required=True)
        score = phases.add_parser("score", help="Score frozen outputs without generation")
        score.add_argument("--score-config", type=Path, required=True)
        score.add_argument("--output-dir", type=Path, required=True)
        report = phases.add_parser("report", help="Rebuild reports; optionally compare a baseline")
        report.add_argument("--output-dir", type=Path, required=True)
        if suite == "codec":
            report.add_argument("--config", type=Path, help="Named runs for Codec comparison")
            continue
        report.add_argument("--baseline", type=Path)
        score.add_argument("--time-budget", type=float)
        run.add_argument("--model", required=True)
        run.add_argument("--base-url", default="http://127.0.0.1:8000")
        run.add_argument(
            "--deployment", type=Path, help="Operator-supplied service identity/settings"
        )
        run.add_argument(
            "--datasets", nargs="+", choices=("llama_questions", "trivia_qa", "web_questions")
        )
        run.add_argument("--tasks", nargs="+", choices=("t2t", "t2s", "s2t", "s2s"))
        selection = run.add_mutually_exclusive_group()
        selection.add_argument("--samples", nargs="+", metavar="DATASET=COUNT")
        selection.add_argument("--selection", type=Path, help="Reuse an existing selection.json")
        run.add_argument("--subset-seed", type=int, help="Selection seed, default 42")
        run.add_argument("--concurrency", nargs="+", type=int, default=[4])
        run.add_argument(
            "--warmup", type=int, help="Requests per cell; default concurrency, 0 disables"
        )
        run.add_argument("--repeats", type=int, default=3)
        run.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
        run.add_argument("--timeout", type=float, default=300)
        run.add_argument("--time-budget", type=float)
        run.add_argument("--sample-rate", type=int, default=24000)
        run.add_argument("--api-key-env", help="API key environment variable; value is never saved")
        for name in ("max-tokens", "seed", "top-k"):
            run.add_argument("--" + name, type=int)
        for name in ("temperature", "top-p", "min-p", "repetition-penalty"):
            run.add_argument("--" + name, type=float)
        run.add_argument(
            "--resume", action="store_true", help="Rerun interrupted cells; keep completed cells"
        )
        run.add_argument(
            "--dry-run", action="store_true", help="Validate inputs and print the plan without HTTP"
        )
    return p


async def run(args):
    if args.suite == "codec":
        from .codec import compare, infer, render, score

        if args.phase == "run":
            result = await asyncio.to_thread(infer, args.config, args.output_dir)
        elif args.phase == "score":
            result = await asyncio.to_thread(score, args.output_dir, args.score_config)
        elif args.config:
            result = compare(
                {
                    name: args.config.resolve().parent / path
                    for name, path in read_json(args.config)["runs"].items()
                },
                args.output_dir,
            )
        else:
            result = render(args.output_dir)
        return {"codec": {"failed": result["status"] != "complete"}}
    if args.phase == "run":
        from .e2e import infer

        return await infer(args)
    from .reporting import report

    if args.phase == "score":
        from .quality import score

        quality = await asyncio.to_thread(
            score, args.output_dir, args.score_config, args.time_budget
        )
        report(args.output_dir)
        return {"quality": {"failed": quality["status"] != "complete"}}
    return report(args.output_dir, args.baseline)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)

    async def execute():
        loop, task = asyncio.get_running_loop(), asyncio.current_task()

        def interrupt(signum, frame):
            loop.call_soon_threadsafe(task.cancel)

        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            return await run(args)
        finally:
            signal.signal(signal.SIGTERM, previous)

    try:
        result = asyncio.run(execute())
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        p.exit(2, f"error: {exc}\n")
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130
    if getattr(args, "dry_run", False):
        print(dumps(result))
        return 0
    print(f"Report: {args.output_dir / 'summary.md'}")
    if any(
        isinstance(v, dict) and (v.get("failed") or v.get("not_attempted")) for v in result.values()
    ):
        return 1
    if args.suite == "e2e":
        summary = read_json(args.output_dir / "summary.json")
        if summary["suite_status"] != "complete":
            return 1
        if args.phase in ("score", "report") and summary.get("quality_status") == "partial":
            return 1
    return 0
