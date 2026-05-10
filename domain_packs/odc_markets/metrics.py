from __future__ import annotations

from typing import Any


def summarize_market_metrics(backtest_result: dict[str, Any]) -> dict[str, Any]:
    """Domain metric adapter for ODC Markets reports and dashboards."""

    return {
        "return_quality": backtest_result.get("total_return"),
        "risk_adjusted_quality": backtest_result.get("sharpe"),
        "path_risk": backtest_result.get("max_drawdown"),
        "activity": backtest_result.get("trades"),
        "exposure": backtest_result.get("mean_abs_exposure"),
    }
