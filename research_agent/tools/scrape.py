from __future__ import annotations

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from research_agent.tools.base import ToolResult


class WebpageScraperTool:
    """Fetch and clean webpage text from HTTP(S) URLs."""

    def __init__(self, timeout_seconds: int = 10, max_chars: int = 12000) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_chars = max_chars

    def run(self, url: str) -> ToolResult:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ToolResult(ok=False, error=f"Blocked non-http URL: {url}")
        try:
            response = requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "ODC-Markets-Research-Agent/0.1"},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()
            text = " ".join(soup.get_text(" ").split())
            return ToolResult(
                ok=True,
                data={
                    "url": url,
                    "title": soup.title.string.strip() if soup.title and soup.title.string else None,
                    "text": text[: self.max_chars],
                },
            )
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc), data={"url": url})
