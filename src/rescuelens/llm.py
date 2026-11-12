"""Shared LLM setup: pick and build the backend that reaches Claude."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rescuelens.config import Settings

if TYPE_CHECKING:
    from rescuelens.verify.backends import LLMBackend

# Models that accept the server-side refusal fallback (`fallbacks="default"`).
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5"}
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def make_backend(settings: Settings, model: str | None = None) -> LLMBackend | None:
    """Return the configured backend, or None when no way to reach Claude is available.

    ``auto`` prefers the API when a key is set, otherwise the local Claude Code CLI.
    """
    import anthropic

    from rescuelens.verify.backends import (
        AnthropicApiBackend,
        ClaudeCodeBackend,
        claude_cli_path,
    )

    model = model or settings.anthropic_model
    choice = settings.llm_backend
    key = settings.anthropic_api_key
    if choice == "api" or (choice == "auto" and key is not None):
        if key is None:
            raise ValueError("RESCUELENS_LLM_BACKEND=api requires ANTHROPIC_API_KEY")
        # SDK retries disabled: the Verifier retries with its own backoff policy.
        client = anthropic.AsyncAnthropic(
            api_key=key.get_secret_value(), max_retries=0, timeout=120.0
        )
        return AnthropicApiBackend(client=client, model=model)
    if choice in ("claude-code", "auto") and claude_cli_path():
        return ClaudeCodeBackend(model=model)
    if choice == "claude-code":
        raise ValueError("RESCUELENS_LLM_BACKEND=claude-code but `claude` is not on PATH")
    return None
