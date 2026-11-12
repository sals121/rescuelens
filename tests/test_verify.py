"""Verification tests with Claude mocked out (no network, no subprocess, no cost)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

from rescuelens.verify.backends import (
    AnthropicApiBackend,
    BackendError,
    BackendName,
    BackendResponse,
    Effort,
    RetryableBackendError,
    build_cli_args,
    parse_cli_stream,
)
from rescuelens.verify.cache import VerdictCache
from rescuelens.verify.client import VerificationError, Verifier, VerifyRequest
from rescuelens.verify.cost import BudgetExceededError, CostTracker, TokenUsage, cost_usd
from rescuelens.verify.schema import Verification

GOOD = Verification(
    is_person=True,
    confidence=0.93,
    evidence="Human figure in red jacket with a cast shadow.",
    posture="lying",
    priority="high",
)
REQ = VerifyRequest(crop_jpeg=b"crop-bytes", context_jpeg=b"context-bytes")


# --------------------------------------------------------------------------- Verifier


class FakeBackend:
    name: BackendName = "claude-code"

    def __init__(self, outcomes: list[Any], model: str = "claude-opus-5-5") -> None:
        self.model = model
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def verify(
        self, *, system: str, images: list[bytes], text: str, effort: Effort
    ) -> BackendResponse:
        self.calls.append({"system": system, "images": images, "text": text, "effort": effort})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, BackendResponse)
        return outcome

    async def write(self, *, system: str, text: str, effort: Effort) -> BackendResponse:
        raise NotImplementedError


def _ok(cost: float = 0.02) -> BackendResponse:
    return BackendResponse(GOOD, "", TokenUsage(1000, 200), cost, "claude-opus-5-5")


def _verifier(
    tmp_path: Path, outcomes: list[Any], *, budget: float = 10.0
) -> tuple[Verifier, FakeBackend]:
    backend = FakeBackend(outcomes)
    v = Verifier(
        backend=backend,
        system_prompt="system",
        prompt_version="v1",
        cache=VerdictCache(tmp_path / "cache"),
        tracker=CostTracker(max_cost_usd=budget),
        max_concurrency=2,
        max_attempts=3,
        max_wait_s=0.01,
    )
    return v, backend


async def test_verify_returns_verdict_and_tracks_cost(tmp_path: Path) -> None:
    v, backend = _verifier(tmp_path, [_ok(0.02)])
    res = await v.verify(REQ)
    assert res.verification == GOOD and not res.cached
    assert v.tracker.spent_usd == pytest.approx(0.02)
    assert v.tracker.usage.input_tokens == 1000
    call = backend.calls[0]
    assert call["images"] == [b"crop-bytes", b"context-bytes"]
    assert call["effort"] == "low"
    assert "yolo" not in call["text"].lower()  # detector score must not leak into the prompt


async def test_second_call_hits_disk_cache(tmp_path: Path) -> None:
    v, backend = _verifier(tmp_path, [_ok()])
    await v.verify(REQ)
    res = await v.verify(REQ)
    assert res.cached and res.cost_usd == 0.0
    assert len(backend.calls) == 1
    assert v.tracker.cache_hits == 1


async def test_cache_key_depends_on_image(tmp_path: Path) -> None:
    v, backend = _verifier(tmp_path, [_ok(), _ok()])
    await v.verify(REQ)
    await v.verify(VerifyRequest(crop_jpeg=b"other", context_jpeg=b"context-bytes"))
    assert len(backend.calls) == 2


async def test_retries_transient_errors(tmp_path: Path) -> None:
    v, backend = _verifier(tmp_path, [RetryableBackendError("rate limit"), _ok()])
    res = await v.verify(REQ)
    assert res.verification.is_person
    assert len(backend.calls) == 2


async def test_gives_up_after_max_attempts(tmp_path: Path) -> None:
    v, backend = _verifier(tmp_path, [RetryableBackendError("overloaded")] * 3)
    with pytest.raises(RetryableBackendError):
        await v.verify(REQ)
    assert len(backend.calls) == 3


async def test_permanent_errors_are_not_retried(tmp_path: Path) -> None:
    v, backend = _verifier(tmp_path, [BackendError("model refused")])
    with pytest.raises(VerificationError, match="refused"):
        await v.verify(REQ)
    assert len(backend.calls) == 1


async def test_budget_cap_blocks_calls(tmp_path: Path) -> None:
    v, backend = _verifier(tmp_path, [_ok()], budget=0.0001)
    with pytest.raises(BudgetExceededError):
        await v.verify(REQ)
    assert backend.calls == []


# --------------------------------------------------------------------------- schema / cost


@pytest.mark.parametrize(
    "payload",
    [
        {**GOOD.model_dump(), "confidence": 1.5},
        {**GOOD.model_dump(), "posture": "flying"},
        {**GOOD.model_dump(), "priority": "urgent"},
        {**GOOD.model_dump(), "extra": "field"},
        {k: v for k, v in GOOD.model_dump().items() if k != "evidence"},
    ],
)
def test_schema_rejects_invalid_payloads(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Verification.model_validate(payload)


def test_cost_matches_price_table() -> None:
    # Opus 5.5: $4 / $20 per MTok
    assert cost_usd("claude-opus-5-5", TokenUsage(1_000_000, 0)) == pytest.approx(4.0)
    assert cost_usd("claude-opus-5-5", TokenUsage(0, 1_000_000)) == pytest.approx(20.0)


def test_unknown_model_price_raises() -> None:
    with pytest.raises(ValueError, match="No price"):
        cost_usd("claude-unknown", TokenUsage(1, 1))


def test_corrupt_cache_entry_is_ignored(tmp_path: Path) -> None:
    cache = VerdictCache(tmp_path)
    key = "ab" + "0" * 62
    (tmp_path / "ab").mkdir()
    (tmp_path / "ab" / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None


# --------------------------------------------------------------------------- claude-code CLI


def test_cli_args_are_locked_down(tmp_path: Path) -> None:
    args = build_cli_args(
        "claude",
        model="claude-opus-5-5",
        system_file=tmp_path / "s.md",
        effort="low",
        schema={"type": "object"},
    )
    assert args[:2] == ["claude", "-p"]
    joined = " ".join(args)
    for flag in ("--tools", "--strict-mcp-config", "--setting-sources", "--no-session-persistence"):
        assert flag in joined
    assert args[args.index("--tools") + 1] == ""
    assert json.loads(args[args.index("--json-schema") + 1]) == {"type": "object"}


def test_parse_cli_stream_takes_final_result() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            "not json",
            json.dumps({"type": "assistant"}),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "structured_output": GOOD.model_dump(),
                    "total_cost_usd": 0.024,
                }
            ),
        ]
    )
    result = parse_cli_stream(stdout)
    assert result["total_cost_usd"] == 0.024
    assert Verification.model_validate(result["structured_output"]) == GOOD


def test_parse_cli_stream_without_result_is_retryable() -> None:
    with pytest.raises(RetryableBackendError):
        parse_cli_stream(json.dumps({"type": "system"}))


# --------------------------------------------------------------------------- Anthropic API


def _api_response(
    parsed: Verification | None = GOOD,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
) -> Any:
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        model=model,
        _request_id="req_test",
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=200,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=None,
        ),
    )


class FakeMessages:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _api_backend(
    outcome: Any, model: str = "claude-opus-5-5"
) -> tuple[AnthropicApiBackend, FakeMessages]:
    messages = FakeMessages(outcome)
    client: Any = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return AnthropicApiBackend(client=client, model=model), messages


async def test_api_backend_parses_and_prices() -> None:
    backend, messages = _api_backend(_api_response())
    res = await backend.verify(
        system="s", images=[b"\xff\xd8a", b"\xff\xd8b"], text="t", effort="low"
    )
    assert res.verification == GOOD
    assert res.cost_usd == pytest.approx(cost_usd("claude-opus-5-5", TokenUsage(1000, 200)))
    call = messages.calls[0]
    assert call["output_format"] is Verification
    assert call["fallbacks"] == "default"  # refusal fallback on Opus 5.5
    assert [b["type"] for b in call["messages"][0]["content"]] == ["image", "image", "text"]


async def test_api_backend_haiku_has_no_fallback() -> None:
    backend, messages = _api_backend(_api_response(model="claude-haiku-5-5"), "claude-haiku-5-5")
    await backend.verify(system="s", images=[], text="t", effort="low")
    assert "fallbacks" not in messages.calls[0]


async def test_api_backend_rate_limit_is_retryable() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=request), body=None
    )
    backend, _ = _api_backend(err)
    with pytest.raises(RetryableBackendError):
        await backend.verify(system="s", images=[], text="t", effort="low")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_api_response(stop_reason="refusal"), "refused"),
        (_api_response(stop_reason="max_tokens"), "truncated"),
        (_api_response(parsed=None), "no structured output"),
    ],
)
async def test_api_backend_unusable_responses(response: Any, message: str) -> None:
    backend, _ = _api_backend(response)
    with pytest.raises(BackendError, match=message):
        await backend.verify(system="s", images=[], text="t", effort="low")
