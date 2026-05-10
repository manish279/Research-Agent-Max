from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AgentSettings:
    """Runtime settings shared by the core graph and domain packs."""

    project_root: Path
    ollama_base_url: str = "http://localhost:11434"
    planner_model: str = "qwen2.5:7b"
    generator_model: str = "qwen2.5-coder:7b"
    critic_model: str = "deepseek-r1:8b"
    embedding_model: str = "nomic-embed-text"
    log_dir: Path = Path("logs")
    memory_dir: Path = Path("memory")
    data_dir: Path = Path("data")
    python_exec_timeout_seconds: int = 12
    allow_live_market_downloads: bool = True

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "AgentSettings":
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        env_file = root / ".env"
        if env_file.exists():
            load_dotenv(env_file)

        return cls(
            project_root=root,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            planner_model=os.getenv("PLANNER_MODEL", "qwen2.5:7b"),
            generator_model=os.getenv("GENERATOR_MODEL", "qwen2.5-coder:7b"),
            critic_model=os.getenv("CRITIC_MODEL", "deepseek-r1:8b"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            log_dir=root / os.getenv("ODC_AGENT_LOG_DIR", "logs"),
            memory_dir=root / os.getenv("ODC_AGENT_MEMORY_DIR", "memory"),
            data_dir=root / os.getenv("ODC_AGENT_DATA_DIR", "data"),
            python_exec_timeout_seconds=int(os.getenv("PYTHON_EXEC_TIMEOUT_SECONDS", "12")),
            allow_live_market_downloads=os.getenv("ALLOW_LIVE_MARKET_DOWNLOADS", "true").lower()
            in {"1", "true", "yes"},
        )

    def ensure_directories(self) -> None:
        for path in (self.log_dir, self.memory_dir, self.data_dir):
            path.mkdir(parents=True, exist_ok=True)
