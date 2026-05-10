from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DomainPack:
    """Domain-specific extension surface for the shared graph kernel."""

    name: str
    description: str
    prompt_dir: Path
    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    risk_limits: dict[str, float] = field(default_factory=dict)
    report_warning: str = "Research-only output. Not investment, legal, medical, or safety advice."

    def prompt(self, name: str) -> str:
        path = self.prompt_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")
