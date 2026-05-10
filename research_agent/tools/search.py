from __future__ import annotations

from research_agent.tools.base import ToolResult


class WebSearchTool:
    """DuckDuckGo search wrapper.

    Network access depends on the runtime environment. Failures are returned as
    tool results so the graph can still produce a transparent report.
    """

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results

    def run(self, query: str) -> ToolResult:
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
            normalized = [
                {
                    "title": item.get("title"),
                    "url": item.get("href") or item.get("url"),
                    "snippet": item.get("body"),
                }
                for item in results
            ]
            return ToolResult(ok=True, data=normalized)
        except Exception as exc:
            return ToolResult(ok=False, data=[], error=str(exc))
