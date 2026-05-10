from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class ExperimentLogger:
    """Persist every important step of a research run.

    Each run gets an immutable-ish directory so strategy ideas, generated code,
    code outputs, backtests, critiques, and reports can be audited later.
    """

    def __init__(self, log_root: Path) -> None:
        self.log_root = log_root
        self.log_root.mkdir(parents=True, exist_ok=True)

    def start_run(self, domain: str, question: str) -> tuple[str, Path]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_{domain}_{uuid4().hex[:8]}"
        run_dir = self.log_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.write_json(run_dir / "run_summary.json", {"run_id": run_id, "domain": domain, "question": question})
        return run_id, run_dir

    def append_event(self, run_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=_json_default) + "\n")

    def write_json(self, path: Path, payload: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return str(path)

    def write_text(self, path: Path, text: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
