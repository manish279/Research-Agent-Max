from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from research_agent.config import AgentSettings


def make_ollama_chat(model: str, settings: AgentSettings, temperature: float = 0.2) -> ChatOllama:
    """Create a local Ollama chat model."""

    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


def invoke_text(
    model: ChatOllama,
    prompt: str,
    system: str | None = None,
    fallback: str | None = None,
) -> str:
    """Invoke an Ollama chat model and normalize the response to text.

    The fallback keeps the graph debuggable when Ollama is not running; production
    deployments should monitor these fallbacks and fail closed if desired.
    """

    messages: list[Any] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    try:
        response = model.invoke(messages)
        content = getattr(response, "content", response)
        return str(content).strip()
    except Exception as exc:  # pragma: no cover - depends on local Ollama service
        if fallback is not None:
            return f"{fallback}\n\n[LLM fallback used because local Ollama call failed: {exc}]"
        raise
