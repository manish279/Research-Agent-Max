from __future__ import annotations

from collections import defaultdict
from typing import Any


class SafeFormatDict(defaultdict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_template(template: str, **context: Any) -> str:
    """Render prompt templates while leaving unknown placeholders visible."""

    values = SafeFormatDict(str)
    values.update({key: _stringify(value) for key, value in context.items()})
    return template.format_map(values)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)
