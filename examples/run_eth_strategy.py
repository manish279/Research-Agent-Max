from __future__ import annotations

import argparse
from pathlib import Path

from domain_packs.odc_markets import get_domain_pack
from research_agent.config import AgentSettings
from research_agent.graph import ResearchGraph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ODC Markets ETH research agent experiment.")
    parser.add_argument("--asset", default="ETH-USD", help="Yahoo Finance symbol, e.g. ETH-USD.")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument(
        "--question",
        default=(
            "Analyze ETH market behavior and generate a simple, auditable strategy hypothesis "
            "that can be backtested for paper-trading research."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    settings = AgentSettings.from_env(project_root)
    graph = ResearchGraph(settings=settings, domain_pack=get_domain_pack())
    result = graph.run(
        question=args.question,
        asset=args.asset,
        lookback_days=args.lookback_days,
        timeframe=args.timeframe,
    )

    run_dir = result.get("artifacts", {}).get("run_dir")
    print(f"Run complete: {result.get('run_id')}")
    print(f"Artifacts: {run_dir}")
    print("\nFinal report preview:\n")
    print(result.get("final_report", "")[:4000])


if __name__ == "__main__":
    main()
