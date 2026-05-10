from __future__ import annotations

from typing import Any


RESEARCH_WARNING = (
    "RESEARCH ONLY / PAPER TRADING WARNING: This report is experimental market research, "
    "not financial advice, not a recommendation, and not approval for live trading."
)


def fallback_report(state: dict[str, Any], domain_warning: str) -> str:
    backtest = state.get("backtest_result", {})
    risk = state.get("risk_report", {})
    critique = state.get("critique", "")
    return f"""# {state.get('asset', 'Asset')} Research Report

{RESEARCH_WARNING}

Domain warning: {domain_warning}

## Research Question

{state.get('question', '')}

## Strategy Idea

{state.get('strategy_idea', 'No strategy idea generated.')}

## Backtest Snapshot

- Total return: {backtest.get('total_return', 'n/a')}
- Sharpe: {backtest.get('sharpe', 'n/a')}
- Max drawdown: {backtest.get('max_drawdown', 'n/a')}
- Trades: {backtest.get('trades', 'n/a')}

## Risk Review

Risk grade: {risk.get('risk_grade', 'n/a')}

{risk.get('warnings', [])}

## Critique

{critique}

## Next Validation Steps

1. Re-run on fresh out-of-sample data.
2. Compare against buy-and-hold and no-trade baselines.
3. Add realistic slippage, fees, funding, liquidity, and latency assumptions.
4. Paper trade before any live consideration.
"""
