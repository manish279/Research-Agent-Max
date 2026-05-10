from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResearchMemory:
    """LlamaIndex-backed memory with a simple JSONL fallback.

    The fallback is intentional: the graph remains testable on machines where
    LlamaIndex/Ollama embeddings are not installed yet, while production runs
    still use a persisted LlamaIndex vector index when dependencies are present.

    Any failure during LlamaIndex initialisation is stored in ``init_warnings``
    so the graph can surface it in the run state rather than silently degrading.
    """

    def __init__(self, persist_dir: Path, embedding_model: str = "nomic-embed-text") -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.jsonl_path = self.persist_dir / "memory.jsonl"
        self._index: Any | None = None
        self._llama_available = False
        self.init_warnings: list[str] = []
        self._init_llama_index()

    def add_text(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "metadata": metadata or {},
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

        if self._llama_available:
            from llama_index.core import Document

            self._index.insert(Document(text=text, metadata=metadata or {}))
            self._index.storage_context.persist(persist_dir=str(self.persist_dir / "llama_index"))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._llama_available and self._index is not None:
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            return [
                {
                    "text": node.node.get_content(),
                    "score": float(getattr(node, "score", 0.0) or 0.0),
                    "metadata": node.node.metadata,
                }
                for node in nodes
            ]

        records = self._read_jsonl()
        query_terms = {term.lower() for term in query.split() if len(term) > 2}
        scored: list[tuple[int, dict[str, Any]]] = []
        for record in records:
            text = record.get("text", "")
            score = sum(1 for term in query_terms if term in text.lower())
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"text": record["text"], "score": score, "metadata": record.get("metadata", {})}
            for score, record in scored[:top_k]
        ]

    def _init_llama_index(self) -> None:
        try:
            from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
            from llama_index.embeddings.ollama import OllamaEmbedding
        except Exception as exc:
            # LlamaIndex not installed — JSONL fallback will be used, nothing to warn about.
            return

        try:
            index_dir = self.persist_dir / "llama_index"
            Settings.embed_model = OllamaEmbedding(model_name=self.embedding_model)
            if index_dir.exists() and any(index_dir.iterdir()):
                storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
                self._index = load_index_from_storage(storage_context)
            else:
                self._index = VectorStoreIndex([])
                self._index.storage_context.persist(persist_dir=str(index_dir))
            self._llama_available = True
        except Exception as exc:
            self._index = None
            self._llama_available = False
            self.init_warnings.append(
                f"LlamaIndex initialisation failed; falling back to keyword search. "
                f"Reason: {exc}"
            )

    def _read_jsonl(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        records = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
