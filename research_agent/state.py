from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Serializable state passed between LangGraph nodes."""

    run_id: str
    domain: str
    question: str
    asset: str
    timeframe: str
    lookback_days: int
    plan: str
    memory_context: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    scraped_pages: list[dict[str, Any]]
    market_data_path: str
    market_summary: dict[str, Any]
    strategy_idea: str
    strategy_code: str
    code_output: dict[str, Any]
    backtest_result: dict[str, Any]
    risk_report: dict[str, Any]
    critique: str
    final_report: str
    artifacts: dict[str, str]
    warnings: list[str]
    errors: list[str]
