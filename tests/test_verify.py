"""Verifier tests with the Anthropic API mocked out (no network, no cost)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

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


def _response(
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
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _fake_client(outcomes: list[Any]) -> tuple[Any, FakeMessages]:
    messages = FakeMessages(outcomes)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def _verifier(
    tmp_path: Path, outcomes: list[Any], *, model: str = "claude-opus-5-5", budget: float = 10.0
) -> tuple[Verifier, FakeMessages]:
    client, messages = _fake_client(outcomes)
    v = Verifier(
        client=client,
        model=model,
        system_prompt="system",
        prompt_version="v1",
        cache=VerdictCache(tmp_path / "cache"),
        tracker=CostTracker(max_cost_usd=budget),
        max_concurrency=2,
    )
    # Make tenacity retries instant.
    v._max_attempts = 3
    return v, messages


REQ = VerifyRequest(crop_jpeg=b"crop-bytes", context_jpeg=b"context-bytes")


def _rate_limit_error() -> anthropic.RateLimitError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(429, request=request, headers={"retry-after": "0"})
    return anthropic.RateLimitError("rate limited", response=response, body=None)


async def test_verify_parses_and_tracks_cost(tmp_path: Path) -> None:
    v, messages = _verifier(tmp_path, [_response()])
    res = await v.verify(REQ)
    assert res.verification == GOOD
    assert not res.cached
    expected = cost_usd("claude-opus-5-5", TokenUsage(1000, 200))
    assert res.cost_usd == pytest.approx(expected)
    assert v.tracker.spent_usd == pytest.approx(expected)

    call = messages.calls[0]
    assert call["output_format"] is Verification
    assert call["fallbacks"] == "default"  # refusal fallback enabled for Opus 5.5
    content = call["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "image", "text"]


async def test_second_call_hits_disk_cache(tmp_path: Path) -> None:
    v, messages = _verifier(tmp_path, [_response()])
    await v.verify(REQ)
    res = await v.verify(REQ)
    assert res.cached and res.cost_usd == 0.0
    assert len(messages.calls) == 1
    assert v.tracker.cache_hits == 1


async def test_cache_key_depends_on_image(tmp_path: Path) -> None:
    v, messages = _verifier(tmp_path, [_response(), _response()])
    await v.verify(REQ)
    await v.verify(VerifyRequest(crop_jpeg=b"other", context_jpeg=b"context-bytes"))
    assert len(messages.calls) == 2


async def test_retries_rate_limit_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    v, messages = _verifier(tmp_path, [_rate_limit_error(), _response()])
    res = await v.verify(REQ)
    assert res.verification.is_person
    assert len(messages.calls) == 2


async def test_gives_up_after_max_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    v, _ = _verifier(tmp_path, [_rate_limit_error() for _ in range(3)])
    with pytest.raises(anthropic.RateLimitError):
        await v.verify(REQ)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_response(stop_reason="refusal"), "refused"),
        (_response(stop_reason="max_tokens"), "truncated"),
        (_response(parsed=None), "no structured output"),
    ],
)
async def test_unusable_responses_raise(tmp_path: Path, response: Any, message: str) -> None:
    v, _ = _verifier(tmp_path, [response])
    with pytest.raises(VerificationError, match=message):
        await v.verify(REQ)


async def test_budget_cap_blocks_calls(tmp_path: Path) -> None:
    v, messages = _verifier(tmp_path, [_response()], budget=0.0001)
    with pytest.raises(BudgetExceededError):
        await v.verify(REQ)
    assert messages.calls == []


async def test_haiku_has_no_fallback_param(tmp_path: Path) -> None:
    v, messages = _verifier(
        tmp_path, [_response(model="claude-haiku-5-5")], model="claude-haiku-5-5"
    )
    await v.verify(REQ)
    assert "fallbacks" not in messages.calls[0]


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


def test_unknown_model_price_raises() -> None:
    with pytest.raises(ValueError, match="No price"):
        cost_usd("claude-unknown", TokenUsage(1, 1))


def test_corrupt_cache_entry_is_ignored(tmp_path: Path) -> None:
    cache = VerdictCache(tmp_path)
    key = "ab" + "0" * 62
    (tmp_path / "ab").mkdir()
    (tmp_path / "ab" / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None


async def _no_sleep(_: float) -> None:
    return None
