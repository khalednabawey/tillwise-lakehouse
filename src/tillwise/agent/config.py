"""Agent settings from environment variables (a local .env file is read if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    openrouter_api_key: str | None
    primary_model: str
    fallback_model: str
    mode: str                 # "auto" (Groq, then OpenRouter), "groq" or "openrouter"
    timeout_seconds: float
    max_steps: int            # tool-calling rounds before the agent must answer


def settings() -> Settings:
    _load_dotenv()
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        primary_model=os.getenv("LLM_PRIMARY_MODEL", "openai/gpt-oss-120b"),
        fallback_model=os.getenv("LLM_FALLBACK_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
        mode=os.getenv("LLM_MODE", "auto").lower(),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
    )
