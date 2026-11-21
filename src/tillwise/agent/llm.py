"""Model router: Groq first, OpenRouter if Groq raises (rate limit, timeout, provider error).

Tools are bound to both models with the same schemas, and every reply is tagged with the
provider and model that actually answered, so the API and the evals can report it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from tillwise.agent.config import Settings, settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _tag(provider: str, model: str) -> RunnableLambda:
    def tag(msg: AIMessage) -> AIMessage:
        msg.response_metadata = {**msg.response_metadata, "provider": provider, "model": model}
        return msg

    return RunnableLambda(tag, name=f"tag_{provider}")


def _groq(cfg: Settings) -> BaseChatModel:
    from langchain_groq import ChatGroq

    retries = 4 if cfg.mode == "groq" else 0  # alone, wait out rate limits; with a fallback, fail over fast
    return ChatGroq(model=cfg.primary_model, api_key=cfg.groq_api_key, temperature=0,
                    timeout=cfg.timeout_seconds, max_retries=retries)


def _openrouter(cfg: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=cfg.fallback_model, api_key=cfg.openrouter_api_key, base_url=OPENROUTER_BASE_URL,
                      temperature=0, timeout=cfg.timeout_seconds * 2, max_retries=1,
                      default_headers={"X-Title": "Tillwise data agent"})


def chat_model(tools: Sequence[Any], cfg: Settings | None = None,
               primary: BaseChatModel | None = None, fallback: BaseChatModel | None = None) -> Runnable:
    """The model arguments exist so tests can pass fakes."""
    cfg = cfg or settings()
    first = (primary or _groq(cfg)).bind_tools(tools) | _tag("groq", cfg.primary_model)
    if cfg.mode == "groq":
        return first
    second = (fallback or _openrouter(cfg)).bind_tools(tools) | _tag("openrouter", cfg.fallback_model)
    if cfg.mode == "openrouter":
        return second
    return first.with_fallbacks([second])
