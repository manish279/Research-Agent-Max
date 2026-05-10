from __future__ import annotations

from pathlib import Path

from research_agent.domain import DomainPack


def get_domain_pack() -> DomainPack:
    """ODC Markets trading-research configuration.

    Keep trading-specific prompts, metrics, and risk limits here. The graph
    kernel should not need to know whether it is researching ETH, equities, or a
    future scientific CosmoTwin domain.
    """

    prompt_dir = Path(__file__).resolve().parent / "prompts"
    return DomainPack(
        name="odc_markets",
        description="Market research, strategy ideation, backtesting, and risk review.",
        prompt_dir=prompt_dir,
        metrics={
            "primary": ["total_return", "sharpe", "max_drawdown", "trades"],
            "risk": ["mean_abs_exposure", "annualized_volatility", "sample_size"],
        },
        risk_limits={
            "max_drawdown_floor": -0.30,
            "min_trades": 3,
            "max_mean_abs_exposure": 1.0,
            "min_rows": 120,
        },
        report_warning=(
            "ODC Markets research-only output. This is for hypothesis generation, "
            "paper trading, and review, not financial advice or live-trading approval."
        ),
    )
