"""Command-line entry point: `uv run rescuelens --help`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from rescuelens import __version__
from rescuelens.config import PROJECT_ROOT, get_settings
from rescuelens.logging import configure_logging, get_logger

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
log = get_logger(__name__)

DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolo26s.pt"


def _default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


@app.callback()
def _main() -> None:
    s = get_settings()
    configure_logging(s.log_level, json=s.log_json)


@app.command()
def version() -> None:
    """Print the RescueLens version."""
    typer.echo(__version__)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8731, reload: bool = False) -> None:
    """Start the FastAPI backend."""
    uvicorn.run("rescuelens.api.app:app", host=host, port=port, reload=reload)


@app.command()
def analyze(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Image folder")],
    weights: Annotated[Path, typer.Option(help="YOLO weights (.pt)")] = DEFAULT_WEIGHTS,
    device: Annotated[str | None, typer.Option(help="cpu | cuda:0 (default: auto)")] = None,
    sahi: Annotated[bool, typer.Option(help="Use SAHI slicing")] = True,
    conf: Annotated[float, typer.Option(help="Detector confidence threshold")] = 0.25,
    llm: Annotated[bool, typer.Option(help="Verify candidates with Claude")] = True,
    geo: Annotated[str, typer.Option(help="auto | exif | simulated")] = "auto",
    limit: Annotated[int | None, typer.Option(help="Only the first N images")] = None,
) -> None:
    """Analyse a folder of drone images and write runs/<run_id>/run.json."""
    from rescuelens.pipeline import AnalysisOptions, run_analysis

    if geo not in ("auto", "exif", "simulated"):
        raise typer.BadParameter("geo must be auto, exif or simulated")
    opts = AnalysisOptions(
        source_dir=source,
        weights=weights,
        device=device or _default_device(),
        use_sahi=sahi,
        confidence=conf,
        llm_enabled=llm,
        geo_mode=geo,  # type: ignore[arg-type]
        limit=limit,
    )

    async def _run() -> None:
        async for ev in run_analysis(opts, get_settings()):
            if ev.type == "frame_done":
                f = ev.data["frame"]
                typer.echo(
                    f"[{ev.data['progress']:>4.0%}] {f['filename']}: "
                    f"{f['n_candidates']} candidates ({f['detect_s']:.1f}s)"
                )
            elif ev.type == "candidate_verified":
                c = ev.data["candidate"]
                v = c["verification"]
                verdict = (
                    f"person={v['is_person']} conf={v['confidence']:.2f} {v['priority']}"
                    if v
                    else c["verify_status"]
                )
                typer.echo(f"   {c['id']} yolo={c['yolo_score']:.2f} -> {verdict}")
            elif ev.type == "report_ready":
                typer.echo("\n" + ev.data["markdown"] + "\n")
            elif ev.type == "run_finished":
                typer.echo(f"Run {ev.data['run_id']}: {ev.data['stats']}")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
