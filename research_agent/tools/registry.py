from __future__ import annotations

from pathlib import Path

from research_agent.config import AgentSettings
from research_agent.tools.backtesting import BacktestTool
from research_agent.tools.market_data import MarketDataTool
from research_agent.tools.python_exec import SafePythonExecutor
from research_agent.tools.scrape import WebpageScraperTool
from research_agent.tools.search import WebSearchTool


def build_default_tools(settings: AgentSettings) -> dict[str, object]:
    """Core tools available to any domain pack."""

    return {
        "web_search": WebSearchTool(max_results=5),
        "web_scraper": WebpageScraperTool(),
        "python_exec": SafePythonExecutor(timeout_seconds=settings.python_exec_timeout_seconds),
        "market_data": MarketDataTool(
            data_dir=Path(settings.data_dir),
            allow_live_downloads=settings.allow_live_market_downloads,
        ),
        "backtest": BacktestTool(),
    }
