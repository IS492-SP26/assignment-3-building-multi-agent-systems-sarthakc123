"""
Main Entry Point
Can be used to run the system or evaluation.

Usage:
  python main.py --mode cli           # Run CLI interface
  python main.py --mode web           # Run web interface
  python main.py --mode evaluate      # Run batch evaluation
  python main.py --mode demo          # End-to-end demo (single query + judge)
  python main.py --mode autogen       # Run the original autogen example
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path


def run_cli():
    """Run CLI interface."""
    from src.ui.cli import main as cli_main
    cli_main()


def run_web():
    """Run Streamlit web interface."""
    import subprocess
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


async def run_evaluation(config_path: str = "config.yaml", test_queries: str = "data/example_queries.json"):
    """Run batch evaluation using SystemEvaluator."""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from src.orchestrator_factory import create_orchestrator
    from src.evaluation.evaluator import SystemEvaluator

    print("Initializing orchestrator...")
    orchestrator = create_orchestrator(config)
    print(f"Orchestrator: {config.get('system', {}).get('orchestrator', 'autogen')}\n")

    evaluator = SystemEvaluator(config, orchestrator=orchestrator)
    print(f"Running evaluation on {test_queries}...")
    report = await evaluator.evaluate_system(test_queries)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    summary = report.get("summary", {})
    scores = report.get("scores", {})
    print(f"Total queries:      {summary.get('total_queries', 0)}")
    print(f"Successful:         {summary.get('successful', 0)}")
    print(f"Failed:             {summary.get('failed', 0)}")
    print(f"Success rate:       {summary.get('success_rate', 0.0):.1%}")
    print(f"Overall avg score:  {scores.get('overall_average', 0.0):.3f}")
    print("\nScores by criterion:")
    for criterion, score in scores.get("by_criterion", {}).items():
        print(f"  {criterion:25s} {score:.3f}")

    if report.get("error_analysis"):
        print("\nError analysis:")
        for line in report["error_analysis"].splitlines():
            print(f"  {line}")
    print(f"\nFull report saved to outputs/")


async def run_demo(config_path: str = "config.yaml", query: str = None):
    """End-to-end demo: query → agents → safety → judge → exported artifacts."""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from src.orchestrator_factory import create_orchestrator
    from src.evaluation.judge import LLMJudge

    if not query:
        query = "What are the key principles of accessible user interface design?"

    print("=" * 70)
    print(" MULTI-AGENT RESEARCH SYSTEM — END-TO-END DEMO")
    print("=" * 70)
    print(f"  Orchestrator : {config.get('system', {}).get('orchestrator', 'autogen')}")
    print(f"  Model        : {config.get('models', {}).get('default', {}).get('name', '?')}")
    print(f"  Topic        : {config.get('system', {}).get('topic', '?')}")
    print(f"  Query        : {query}\n")

    # 1. Run orchestrator
    print("Step 1/3: running multi-agent workflow...")
    orchestrator = create_orchestrator(config)
    result = orchestrator.process_query(query)

    md = result.get("metadata", {}) or {}
    print(f"  ✓ {md.get('num_messages', 0)} messages exchanged")
    print(f"  ✓ {md.get('num_sources', 0)} sources gathered")
    print(f"  ✓ Agents: {', '.join(md.get('agents_involved', []))}")

    # 2. Run LLM judge
    print("\nStep 2/3: running LLM-as-a-Judge...")
    judge = LLMJudge(config)
    evaluation = await judge.evaluate(
        query=query,
        response=result.get("response", ""),
        sources=md.get("sources", []),
    )
    print(f"  ✓ Overall score: {evaluation.get('overall_score', 0.0):.3f}")
    for crit, score in evaluation.get("criterion_scores", {}).items():
        print(f"      {crit:20s} {score.get('score', 0.0):.3f}")

    # 3. Export artifacts
    print("\nStep 3/3: exporting artifacts...")
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    session_path = outdir / f"session_{ts}.json"
    answer_path = outdir / f"answer_{ts}.md"
    judge_path = outdir / f"judge_{ts}.json"

    full_session = {**result, "evaluation": evaluation}
    with open(session_path, "w") as f:
        json.dump(full_session, f, indent=2, default=str)

    answer_md = _build_answer_markdown(query, result, evaluation, config)
    with open(answer_path, "w") as f:
        f.write(answer_md)

    with open(judge_path, "w") as f:
        json.dump(evaluation, f, indent=2, default=str)

    print(f"  ✓ {session_path}")
    print(f"  ✓ {answer_path}")
    print(f"  ✓ {judge_path}")

    print("\n" + "=" * 70)
    print(" FINAL ANSWER (first 500 chars)")
    print("=" * 70)
    print((result.get("response") or "")[:500])
    print("\n" + "=" * 70)


def _build_answer_markdown(query: str, result: dict, evaluation: dict, config: dict) -> str:
    md = []
    md.append("# Research Answer\n")
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    md.append(f"**Orchestrator:** `{config.get('system', {}).get('orchestrator', 'autogen')}`\n")
    md.append(f"## Query\n{query}\n")
    md.append(f"## Answer\n{result.get('response', '')}\n")
    metadata = result.get("metadata", {}) or {}
    sources = metadata.get("sources") or []
    if sources:
        md.append("## Sources\n")
        for i, s in enumerate(sources, 1):
            url = s.get("url", "") if isinstance(s, dict) else ""
            title = s.get("title", "") if isinstance(s, dict) else ""
            md.append(f"{i}. {title or url}" + (f" — <{url}>" if url and title else ""))
        md.append("")
    md.append("## Evaluation\n")
    md.append(f"- **Overall score:** {evaluation.get('overall_score', 0.0):.3f}")
    for crit, score in evaluation.get("criterion_scores", {}).items():
        md.append(f"- **{crit}:** {score.get('score', 0.0):.3f} — {(score.get('reasoning') or '')[:200]}")
    md.append("")
    events = metadata.get("safety_events", [])
    if events:
        md.append("## Safety Events\n")
        for ev in events:
            cats = ev.get("policy_categories") or ["none"]
            md.append(f"- **{ev.get('type', '').upper()}**: action=`{ev.get('action', '')}` categories=`{', '.join(cats)}`")
    return "\n".join(md)


def run_autogen():
    """Run the original autogen example script."""
    import subprocess
    print("Running AutoGen example...")
    subprocess.run([sys.executable, "example_autogen.py"])


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant")
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "evaluate", "demo", "autogen"],
        default="demo",
        help="Mode to run (default: demo)"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--query",
        default=None,
        help="Override the default demo query (only used with --mode demo)",
    )
    parser.add_argument(
        "--queries",
        default="data/example_queries.json",
        help="Test queries JSON for --mode evaluate (default: data/example_queries.json)",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation(args.config, args.queries))
    elif args.mode == "demo":
        asyncio.run(run_demo(args.config, args.query))
    elif args.mode == "autogen":
        run_autogen()


if __name__ == "__main__":
    main()
