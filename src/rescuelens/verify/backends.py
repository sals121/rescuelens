"""How RescueLens reaches Claude.

Two interchangeable backends:

* ``claude-code`` - runs the Claude Code CLI headless (``claude -p``), authenticated with the
  user's Claude subscription. No API key needed. Reported cost is the CLI's API-equivalent
  list price (``total_cost_usd``), not an actual charge.
* ``api`` - the Anthropic Messages API via the official SDK (needs ``ANTHROPIC_API_KEY``).

Both return the same ``BackendResponse`` so caching, retries and budgeting live in one place.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import anthropic

from rescuelens.llm import FALLBACK_BETA, FALLBACK_MODELS
from rescuelens.logging import get_logger
from rescuelens.verify.cost import TokenUsage, cost_usd
from rescuelens.verify.schema import Verification

log = get_logger(__name__)

BackendName = Literal["claude-code", "api"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class BackendError(RuntimeError):
    """Non-retryable failure (bad request, refusal, unusable output)."""


class RetryableBackendError(RuntimeError):
    """Transient failure worth retrying (rate limit, overload, network, CLI crash)."""


@dataclass(frozen=True, slots=True)
class BackendResponse:
    verification: Verification | None  # None for free-text calls
    text: str
    usage: TokenUsage
    cost_usd: float
    model: str


class LLMBackend(Protocol):
    name: BackendName
    model: str

    async def verify(
        self, *, system: str, images: list[bytes], text: str, effort: Effort
    ) -> BackendResponse: ...

    async def write(self, *, system: str, text: str, effort: Effort) -> BackendResponse: ...


def _b64_image(data: bytes) -> dict[str, Any]:
    media = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media,
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }


# --------------------------------------------------------------------------- claude-code


def claude_cli_path() -> str | None:
    return shutil.which("claude")


def build_cli_args(
    exe: str, *, model: str, system_file: Path, effort: Effort, schema: dict[str, Any] | None
) -> list[str]:
    args = [
        exe,
        "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        "--effort", effort,
        "--system-prompt-file", str(system_file),
        "--tools", "",  # pure inference: no file, shell or web tools
        "--strict-mcp-config",  # ignore the user's MCP servers
        "--setting-sources", "",  # ignore user/project settings, hooks and plugins
        "--no-session-persistence",
    ]  # fmt: skip
    if schema is not None:
        args += ["--json-schema", json.dumps(schema)]
    return args


def parse_cli_stream(stdout: str) -> dict[str, Any]:
    """Return the final ``result`` event of a stream-json transcript."""
    result: dict[str, Any] | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
    if result is None:
        raise RetryableBackendError("claude CLI produced no result event")
    return result


def _usage_from_cli(result: dict[str, Any]) -> TokenUsage:
    u = result.get("usage") or {}
    return TokenUsage(
        input_tokens=int(u.get("input_tokens") or 0),
        output_tokens=int(u.get("output_tokens") or 0),
        cache_creation_input_tokens=int(u.get("cache_creation_input_tokens") or 0),
        cache_read_input_tokens=int(u.get("cache_read_input_tokens") or 0),
    )


_RETRYABLE_HINTS = ("rate limit", "rate_limit", "overloaded", "529", "timeout", "temporarily")


class ClaudeCodeBackend:
    name: BackendName = "claude-code"

    def __init__(self, *, model: str, exe: str | None = None, timeout_s: float = 180.0) -> None:
        found = exe or claude_cli_path()
        if not found:
            raise RuntimeError("Claude Code CLI (`claude`) not found on PATH")
        self._exe = found
        self.model = model
        self._timeout = timeout_s
        self._workdir = Path(tempfile.mkdtemp(prefix="rescuelens-cc-"))  # neutral cwd

    async def _run(
        self,
        system: str,
        content: list[dict[str, Any]],
        effort: Effort,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        sys_file = self._workdir / f"system-{abs(hash(system))}.md"
        if not sys_file.exists():
            sys_file.write_text(system, encoding="utf-8")
        payload = json.dumps(
            {"type": "user", "message": {"role": "user", "content": content}}
        ).encode()
        args = build_cli_args(
            self._exe, model=self.model, system_file=sys_file, effort=effort, schema=schema
        )
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=self._workdir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(payload + b"\n"), self._timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise RetryableBackendError(f"claude CLI timed out after {self._timeout}s") from None

        stdout, stderr = out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
        if proc.returncode != 0 and '"type":"result"' not in stdout.replace(" ", ""):
            msg = (stderr or stdout).strip()[-500:]
            if any(h in msg.lower() for h in _RETRYABLE_HINTS):
                raise RetryableBackendError(msg)
            raise BackendError(f"claude CLI exited {proc.returncode}: {msg}")

        result = parse_cli_stream(stdout)
        if result.get("is_error") or result.get("subtype") != "success":
            msg = str(result.get("result") or result.get("subtype"))
            if any(h in msg.lower() for h in _RETRYABLE_HINTS):
                raise RetryableBackendError(msg)
            raise BackendError(f"claude CLI error: {msg}")
        return result

    async def verify(
        self, *, system: str, images: list[bytes], text: str, effort: Effort
    ) -> BackendResponse:
        content = [*(_b64_image(i) for i in images), {"type": "text", "text": text}]
        result = await self._run(system, content, effort, Verification.model_json_schema())
        raw = result.get("structured_output")
        if raw is None:
            raise BackendError("claude CLI returned no structured_output")
        return BackendResponse(
            verification=Verification.model_validate(raw),
            text=str(result.get("result", "")),
            usage=_usage_from_cli(result),
            cost_usd=float(result.get("total_cost_usd") or 0.0),
            model=self.model,
        )

    async def write(self, *, system: str, text: str, effort: Effort) -> BackendResponse:
        result = await self._run(system, [{"type": "text", "text": text}], effort, None)
        return BackendResponse(
            verification=None,
            text=str(result.get("result", "")),
            usage=_usage_from_cli(result),
            cost_usd=float(result.get("total_cost_usd") or 0.0),
            model=self.model,
        )


# --------------------------------------------------------------------------- api

_API_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,  # includes APITimeoutError
)


class AnthropicApiBackend:
    name: BackendName = "api"

    def __init__(self, *, client: anthropic.AsyncAnthropic, model: str) -> None:
        self._client = client
        self.model = model

    def _extra(self) -> dict[str, Any]:
        if self.model in FALLBACK_MODELS:  # server-side refusal fallback
            return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return {}

    @staticmethod
    def _usage(response: Any) -> TokenUsage:  # noqa: ANN401 - BetaMessage or ParsedBetaMessage
        return TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
        )

    async def verify(
        self, *, system: str, images: list[bytes], text: str, effort: Effort
    ) -> BackendResponse:
        content: list[Any] = [*(_b64_image(i) for i in images), {"type": "text", "text": text}]
        try:
            response = await self._client.beta.messages.parse(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_format=Verification,
                output_config={"effort": effort},
                **self._extra(),
            )
        except _API_RETRYABLE as exc:
            raise RetryableBackendError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise BackendError(str(exc)) from exc

        usage = self._usage(response)
        served_by = str(response.model)
        if response.stop_reason == "refusal":
            raise BackendError(f"model refused (request {response._request_id})")
        if response.stop_reason == "max_tokens":
            raise BackendError("response truncated at max_tokens")
        if response.parsed_output is None:
            raise BackendError("model returned no structured output")
        return BackendResponse(
            verification=response.parsed_output,
            text="",
            usage=usage,
            cost_usd=cost_usd(served_by, usage),
            model=served_by,
        )

    async def write(self, *, system: str, text: str, effort: Effort) -> BackendResponse:
        try:
            response = await self._client.beta.messages.create(
                model=self.model,
                max_tokens=8000,
                system=system,
                messages=[{"role": "user", "content": text}],
                output_config={"effort": effort},
                **self._extra(),
            )
        except _API_RETRYABLE as exc:
            raise RetryableBackendError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise BackendError(str(exc)) from exc
        usage = self._usage(response)
        if response.stop_reason in ("refusal", "max_tokens"):
            raise BackendError(f"report stopped: {response.stop_reason}")
        body = "".join(b.text for b in response.content if b.type == "text")
        return BackendResponse(
            None, body, usage, cost_usd(str(response.model), usage), str(response.model)
        )
