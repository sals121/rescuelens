"""In-process job runner: one analysis at a time (one GPU), events fanned out over SSE.

Every event is kept, so a client that connects late (or reconnects) replays the run from
the start and then follows it live.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from rescuelens.logging import get_logger
from rescuelens.pipeline import AnalysisOptions, Event, run_analysis

log = get_logger(__name__)

Runner = Callable[[AnalysisOptions], AsyncIterator[Event]]


class BusyError(RuntimeError):
    """Another analysis is still running."""


@dataclass
class Job:
    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    _changed: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._changed:
            self.events.append(event)
            self._changed.notify_all()

    async def finish(self) -> None:
        async with self._changed:
            self.done = True
            self._changed.notify_all()

    def _has_news(self, seen: int) -> bool:
        return len(self.events) > seen or self.done

    async def follow(self, start: int = 0) -> AsyncIterator[dict[str, Any]]:
        """Yield every event from `start`, then new ones as they arrive, until the job ends."""
        i = start
        while True:
            async with self._changed:
                await self._changed.wait_for(functools.partial(self._has_news, i))
                batch, finished = self.events[i:], self.done
            for ev in batch:
                yield ev
            i += len(batch)
            if finished and i >= len(self.events):
                return


class JobManager:
    def __init__(self, runner: Runner) -> None:
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._active: asyncio.Task[None] | None = None

    @property
    def busy(self) -> bool:
        return self._active is not None and not self._active.done()

    def get(self, run_id: str) -> Job | None:
        return self._jobs.get(run_id)

    @property
    def active_run_id(self) -> str | None:
        return next((j.run_id for j in self._jobs.values() if not j.done), None)

    def start(self, opts: AnalysisOptions) -> Job:
        if self.busy:
            raise BusyError("an analysis is already running")
        assert opts.run_id is not None, "the caller assigns the run id"
        job = Job(run_id=opts.run_id)
        self._jobs[job.run_id] = job
        self._active = asyncio.create_task(self._drive(job, opts))
        return job

    async def _drive(self, job: Job, opts: AnalysisOptions) -> None:
        try:
            async for ev in self._runner(opts):
                await job.publish(ev.model_dump())
        except Exception as exc:  # surfaced to the client as an event, and logged
            log.exception("analysis_failed", run_id=job.run_id)
            await job.publish(
                {"type": "error", "data": {"run_id": job.run_id, "message": f"{exc}"}}
            )
        finally:
            await job.finish()


def default_runner(settings_factory: Callable[[], Any]) -> Runner:
    def _run(opts: AnalysisOptions) -> AsyncIterator[Event]:
        return run_analysis(opts, settings_factory())

    return _run
