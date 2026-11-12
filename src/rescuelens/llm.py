"""Shared Anthropic client setup."""

from __future__ import annotations

import anthropic

from rescuelens.config import Settings

# Models that accept the server-side refusal fallback (`fallbacks="default"`).
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5"}
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def make_client(settings: Settings) -> anthropic.AsyncAnthropic:
    """Async client. SDK retries are disabled: callers retry with their own backoff policy."""
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    return anthropic.AsyncAnthropic(api_key=key, max_retries=0, timeout=120.0)
