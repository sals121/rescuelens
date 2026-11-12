"""Token accounting and a hard spending cap for Claude calls.

Prices are USD per million tokens, Anthropic first-party API list prices (Nov 2026).
Cache writes are billed at 1.25x input and cache reads at 0.1x input.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

# model id -> (input $/MTok, output $/MTok)
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.00, 40.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10


class BudgetExceededError(RuntimeError):
    """Raised before a call that could push spend past the configured cap."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


def price_for(model: str) -> tuple[float, float]:
    try:
        return PRICES_PER_MTOK[model]
    except KeyError:
        raise ValueError(
            f"No price configured for model {model!r}; add it to PRICES_PER_MTOK."
        ) from None


def cost_usd(model: str, usage: TokenUsage) -> float:
    inp, out = price_for(model)
    return (
        usage.input_tokens * inp
        + usage.cache_creation_input_tokens * inp * CACHE_WRITE_MULT
        + usage.cache_read_input_tokens * inp * CACHE_READ_MULT
        + usage.output_tokens * out
    ) / 1_000_000


@dataclass
class CostTracker:
    """Thread-safe running total of tokens and dollars, with a hard cap."""

    max_cost_usd: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    spent_usd: float = 0.0
    calls: int = 0
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check_budget(self, reserve_usd: float = 0.0) -> None:
        with self._lock:
            if self.spent_usd + reserve_usd > self.max_cost_usd:
                raise BudgetExceededError(
                    f"LLM budget of ${self.max_cost_usd:.2f} reached "
                    f"(spent ${self.spent_usd:.4f}). Raise RESCUELENS_MAX_COST_USD to continue."
                )

    def record(self, model: str, usage: TokenUsage) -> float:
        call_cost = cost_usd(model, usage)
        with self._lock:
            self.usage = self.usage + usage
            self.spent_usd += call_cost
            self.calls += 1
        return call_cost

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1
