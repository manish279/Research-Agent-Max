from __future__ import annotations

from typing import Any


DEFAULT_TRADING_RISK_LIMITS = {
    "max_drawdown_floor": -0.25,
    "min_trades": 3,
    "max_mean_abs_exposure": 1.0,
    "min_rows": 120,
}


class TradingRiskEvaluator:
    """Evaluate strategy research before it reaches the final report."""

    def __init__(self, limits: dict[str, float] | None = None) -> None:
        self.limits = {**DEFAULT_TRADING_RISK_LIMITS, **(limits or {})}

    def evaluate(self, backtest_result: dict[str, Any]) -> dict[str, Any]:
        checks = []
        checks.append(
            _check(
                "drawdown_limit",
                backtest_result.get("max_drawdown", 0.0) >= self.limits["max_drawdown_floor"],
                f"Max drawdown should be above {self.limits['max_drawdown_floor']:.0%}.",
            )
        )
        checks.append(
            _check(
                "minimum_trade_count",
                backtest_result.get("trades", 0) >= self.limits["min_trades"],
                f"Backtest should include at least {int(self.limits['min_trades'])} position changes.",
            )
        )
        checks.append(
            _check(
                "exposure_limit",
                backtest_result.get("mean_abs_exposure", 0.0) <= self.limits["max_mean_abs_exposure"],
                f"Mean absolute exposure should be <= {self.limits['max_mean_abs_exposure']:.2f}.",
            )
        )
        checks.append(
            _check(
                "sample_size",
                backtest_result.get("rows", 0) >= self.limits["min_rows"],
                f"Backtest should have at least {int(self.limits['min_rows'])} observations.",
            )
        )
        passed = all(item["passed"] for item in checks)
        return {
            "passed": passed,
            "checks": checks,
            "risk_grade": "research-pass" if passed else "needs-review",
            "warnings": [
                "Paper-trading/research-only result. Do not deploy live without out-of-sample validation.",
                "Backtest ignores exchange outages, liquidity constraints, funding, slippage regimes, and tax impacts.",
            ],
        }


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}
