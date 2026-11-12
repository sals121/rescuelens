"""Verify detection candidates with Claude vision: caching, retries and a cost cap."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rescuelens.logging import get_logger
from rescuelens.verify.backends import BackendError, Effort, LLMBackend, RetryableBackendError
from rescuelens.verify.cache import CachedVerdict, VerdictCache, cache_key
from rescuelens.verify.cost import CostTracker, TokenUsage, cost_usd
from rescuelens.verify.schema import Verification

log = get_logger(__name__)

# Rough upper bound per call, used to stop *before* overshooting the budget.
_RESERVE = TokenUsage(input_tokens=3_000, output_tokens=1_000)


class VerificationError(RuntimeError):
    """The model could not produce a usable verdict (refusal, truncation, invalid output)."""


@dataclass(frozen=True, slots=True)
class VerifyRequest:
    crop_jpeg: bytes
    context_jpeg: bytes
    note: str = ""  # optional extra text; must not leak the detector score


@dataclass(frozen=True, slots=True)
class VerifyResult:
    verification: Verification
    cached: bool
    cost_usd: float
    latency_s: float
    model: str
    # what the call cost when it was first made (equal to cost_usd/latency_s unless cached)
    source_cost_usd: float = 0.0
    source_latency_s: float = 0.0


def load_prompt(prompts_dir: Path, name: str, version: str) -> str:
    return (prompts_dir / f"{name}_{version}.md").read_text(encoding="utf-8").strip()


class Verifier:
    def __init__(
        self,
        *,
        backend: LLMBackend,
        system_prompt: str,
        prompt_version: str,
        cache: VerdictCache,
        tracker: CostTracker,
        max_concurrency: int = 4,
        effort: Effort = "low",
        max_attempts: int = 5,
        max_wait_s: float = 30.0,
    ) -> None:
        self.backend = backend
        self._system = system_prompt
        self.prompt_version = prompt_version
        self._cache = cache
        self.tracker = tracker
        self._sem = asyncio.Semaphore(max_concurrency)
        self._effort: Effort = effort
        self._max_attempts = max_attempts
        self._max_wait = max_wait_s
        try:
            self._reserve_usd = cost_usd(backend.model, _RESERVE)
        except ValueError:
            self._reserve_usd = 0.0

    @staticmethod
    def user_text(req: VerifyRequest) -> str:
        text = "Image 1 is the CROP, image 2 is the CONTEXT. Verify the outlined object."
        return f"{text}\n{req.note}" if req.note else text

    async def verify(self, req: VerifyRequest) -> VerifyResult:
        text = self.user_text(req)
        key = cache_key(
            model=self.backend.model,
            prompt_version=self.prompt_version,
            images=[req.crop_jpeg, req.context_jpeg],
            text=text,
        )
        if (hit := self._cache.get(key)) is not None:
            self.tracker.record_cache_hit()
            return VerifyResult(
                hit.verification,
                cached=True,
                cost_usd=0.0,
                latency_s=0.0,
                model=hit.model,
                source_cost_usd=hit.cost_usd,
                source_latency_s=hit.latency_s,
            )

        async with self._sem:
            self.tracker.check_budget(self._reserve_usd)
            start = time.perf_counter()
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(RetryableBackendError),
                    wait=wait_exponential_jitter(initial=1, max=self._max_wait),
                    stop=stop_after_attempt(self._max_attempts),
                    reraise=True,
                ):
                    with attempt:
                        response = await self.backend.verify(
                            system=self._system,
                            images=[req.crop_jpeg, req.context_jpeg],
                            text=text,
                            effort=self._effort,
                        )
            except BackendError as exc:
                raise VerificationError(str(exc)) from exc
            latency = time.perf_counter() - start

        self.tracker.record_cost(response.usage, response.cost_usd)
        if response.verification is None:
            raise VerificationError("backend returned no verdict")

        self._cache.put(
            key,
            CachedVerdict(
                verification=response.verification,
                model=response.model,
                prompt_version=self.prompt_version,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=response.cost_usd,
                latency_s=latency,
            ),
        )
        log.debug(
            "verified",
            is_person=response.verification.is_person,
            cost_usd=round(response.cost_usd, 5),
            latency_s=round(latency, 2),
        )
        return VerifyResult(
            response.verification,
            cached=False,
            cost_usd=response.cost_usd,
            latency_s=latency,
            model=response.model,
            source_cost_usd=response.cost_usd,
            source_latency_s=latency,
        )
