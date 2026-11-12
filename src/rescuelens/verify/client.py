"""Verify detection candidates with Claude vision, with caching, retries and a cost cap."""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anthropic
from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rescuelens.llm import FALLBACK_BETA, FALLBACK_MODELS
from rescuelens.logging import get_logger
from rescuelens.verify.cache import CachedVerdict, VerdictCache, cache_key
from rescuelens.verify.cost import CostTracker, TokenUsage, cost_usd
from rescuelens.verify.schema import Verification

log = get_logger(__name__)

Effort = Literal["low", "medium", "high", "xhigh", "max"]


_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,  # includes APITimeoutError
)

# Rough upper bound per call, used to stop *before* overshooting the budget.
_RESERVE_INPUT_TOKENS = 2_500
_RESERVE_OUTPUT_TOKENS = 1_500


class VerificationError(RuntimeError):
    """The model could not produce a usable verdict (refusal, truncation, invalid output)."""


@dataclass(frozen=True, slots=True)
class VerifyRequest:
    crop_jpeg: bytes
    context_jpeg: bytes
    note: str = ""  # optional extra text (e.g. object size); must not leak the detector score


@dataclass(frozen=True, slots=True)
class VerifyResult:
    verification: Verification
    cached: bool
    cost_usd: float
    latency_s: float
    model: str


def load_prompt(prompts_dir: Path, version: str) -> str:
    return (prompts_dir / f"verify_{version}.md").read_text(encoding="utf-8").strip()


def _image_block(data: bytes) -> anthropic.types.beta.BetaImageBlockParam:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }


class Verifier:
    def __init__(
        self,
        *,
        client: anthropic.AsyncAnthropic,
        model: str,
        system_prompt: str,
        prompt_version: str,
        cache: VerdictCache,
        tracker: CostTracker,
        max_concurrency: int = 4,
        effort: Effort = "low",
        max_attempts: int = 5,
    ) -> None:
        self._client = client
        self.model = model
        self._system = system_prompt
        self.prompt_version = prompt_version
        self._cache = cache
        self.tracker = tracker
        self._sem = asyncio.Semaphore(max_concurrency)
        self._effort: Effort = effort
        self._max_attempts = max_attempts
        reserve = TokenUsage(_RESERVE_INPUT_TOKENS, _RESERVE_OUTPUT_TOKENS)
        self._reserve_usd = cost_usd(model, reserve)

    def _user_text(self, req: VerifyRequest) -> str:
        text = "Image 1 is the CROP, image 2 is the CONTEXT. Verify the outlined object."
        return f"{text}\n{req.note}" if req.note else text

    async def verify(self, req: VerifyRequest) -> VerifyResult:
        text = self._user_text(req)
        key = cache_key(
            model=self.model,
            prompt_version=self.prompt_version,
            images=[req.crop_jpeg, req.context_jpeg],
            text=text,
        )
        if (hit := self._cache.get(key)) is not None:
            self.tracker.record_cache_hit()
            return VerifyResult(
                hit.verification, cached=True, cost_usd=0.0, latency_s=0.0, model=hit.model
            )

        async with self._sem:
            self.tracker.check_budget(self._reserve_usd)
            start = time.perf_counter()
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(_RETRYABLE),
                wait=wait_exponential_jitter(initial=1, max=30),
                stop=stop_after_attempt(self._max_attempts),
                reraise=True,
            ):
                with attempt:
                    response = await self._call(req, text)
            latency = time.perf_counter() - start

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
        )
        served_by = response.model
        call_cost = self.tracker.record(served_by, usage)

        if response.stop_reason == "refusal":
            raise VerificationError(f"model refused (request {response._request_id})")
        if response.stop_reason == "max_tokens":
            raise VerificationError("response truncated at max_tokens")
        parsed = response.parsed_output
        if parsed is None:
            raise VerificationError("model returned no structured output")

        self._cache.put(
            key,
            CachedVerdict(
                verification=parsed,
                model=served_by,
                prompt_version=self.prompt_version,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
        )
        log.debug(
            "verified",
            is_person=parsed.is_person,
            cost_usd=round(call_cost, 5),
            latency_s=round(latency, 2),
        )
        return VerifyResult(
            parsed, cached=False, cost_usd=call_cost, latency_s=latency, model=served_by
        )

    async def _call(self, req: VerifyRequest, text: str) -> ParsedBetaMessage[Verification]:
        content: list[anthropic.types.beta.BetaContentBlockParam] = [
            _image_block(req.crop_jpeg),
            _image_block(req.context_jpeg),
            {"type": "text", "text": text},
        ]
        if self.model in FALLBACK_MODELS:
            return await self._client.beta.messages.parse(
                model=self.model,
                max_tokens=4096,
                system=self._system,
                messages=[{"role": "user", "content": content}],
                output_format=Verification,
                output_config={"effort": self._effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        return await self._client.beta.messages.parse(
            model=self.model,
            max_tokens=4096,
            system=self._system,
            messages=[{"role": "user", "content": content}],
            output_format=Verification,
            output_config={"effort": self._effort},
        )
