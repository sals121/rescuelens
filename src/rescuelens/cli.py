"""Command-line entry point: `uv run rescuelens --help`."""

from __future__ import annotations

import asyncio
import dataclasses
import json
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


data_app = typer.Typer(no_args_is_help=True, help="Download and prepare the HERIDAL dataset.")
app.add_typer(data_app, name="data")


@data_app.command("download")
def data_download(
    dest: Annotated[Path | None, typer.Option(help="Archive path")] = None,
) -> None:
    """Download HERIDAL (8.3 GB, resumable) and verify its md5."""
    from rescuelens.data.download import HERIDAL_MD5, HERIDAL_SIZE, HERIDAL_URL, download

    dest = dest or get_settings().data_dir / "raw" / "heridal_keras_retinanet_voc.zip"
    download(HERIDAL_URL, dest, expected_md5=HERIDAL_MD5, expected_size=HERIDAL_SIZE)
    typer.echo(f"OK {dest} (md5 {HERIDAL_MD5})")


@data_app.command("prepare")
def data_prepare(
    archive: Annotated[Path | None, typer.Option(help="HERIDAL zip")] = None,
    out: Annotated[Path | None, typer.Option(help="Output directory")] = None,
    tile: int = 640,
    overlap: float = 0.2,
    neg_keep: Annotated[float, typer.Option(help="Share of empty tiles kept (train/val)")] = 0.05,
    seed: int = 42,
    workers: Annotated[int | None, typer.Option(help="Worker processes")] = None,
) -> None:
    """Extract the archive and build YOLO tiles + full-resolution test frames."""
    from rescuelens.data.download import safe_extract
    from rescuelens.data.heridal import PrepareConfig, prepare_heridal

    data_dir = get_settings().data_dir
    archive = archive or data_dir / "raw" / "heridal_keras_retinanet_voc.zip"
    out = out or data_dir / "heridal"
    raw = out / "raw"
    if not any(raw.rglob("ImageSets")):
        typer.echo(f"extracting {archive} -> {raw}")
        safe_extract(archive, raw)
    cfg = PrepareConfig(tile=tile, overlap=overlap, neg_keep=neg_keep, seed=seed)
    if workers is not None:
        cfg = dataclasses.replace(cfg, workers=workers)
    manifest = prepare_heridal(raw, out, cfg)
    typer.echo(json.dumps(manifest["splits"], indent=1))


@app.command()
def train(
    data: Annotated[Path | None, typer.Option(help="dataset.yaml")] = None,
    weights: Annotated[Path, typer.Option(help="Starting checkpoint")] = DEFAULT_WEIGHTS,
    out: Annotated[Path, typer.Option(help="Where to copy the best checkpoint")] = (
        PROJECT_ROOT / "weights" / "rescuelens-yolo26s-heridal.pt"
    ),
    epochs: int = 100,
    batch: int = 32,
    device: str = "0",
    name: str = "heridal-yolo26s",
) -> None:
    """Fine-tune YOLO on HERIDAL tiles, then score the best checkpoint on the test tiles."""
    from rescuelens.train import TrainConfig, train_detector

    s = get_settings()
    cfg = TrainConfig(
        data=data or s.data_dir / "heridal" / "tiles" / "dataset.yaml",
        weights=weights,
        out_weights=out,
        project=s.runs_dir / "train",
        name=name,
        epochs=epochs,
        batch=batch,
        device=device,
    )
    summary = train_detector(cfg)
    typer.echo(json.dumps(summary["test_tiles"], indent=1))


eval_app = typer.Typer(no_args_is_help=True, help="Evaluate on the full-resolution test set.")
app.add_typer(eval_app, name="eval")


def _results_dir(out: Path | None) -> Path:
    from datetime import date

    return out or get_settings().results_dir / date.today().isoformat()


def _test_dir() -> Path:
    return get_settings().data_dir / "heridal"


@eval_app.command("detector")
def eval_detector(
    weights: Annotated[list[Path], typer.Option(help="One or more checkpoints")],
    conf: Annotated[float, typer.Option(help="Operating threshold for P/R/F1")] = 0.25,
    device: Annotated[str | None, typer.Option(help="cpu | cuda:0")] = None,
    out: Annotated[Path | None, typer.Option(help="Results directory")] = None,
) -> None:
    """Score each checkpoint on the 101 test frames, with and without SAHI."""
    from rescuelens.data.heridal import ground_truth
    from rescuelens.eval.detection import load_preds, predict_frames, save_preds, score_run

    out = _results_dir(out)
    gts = ground_truth(_test_dir())
    frames = _test_dir() / "test_full" / "images"
    rows: list[dict[str, object]] = []
    for w in weights:
        for sahi in (True, False):
            tag = f"{w.stem}_{'sahi' if sahi else 'full'}"
            path = out / "preds" / f"{tag}.json"
            meta = load_preds(path)[1] if path.exists() else {}
            if "s_per_frame" in meta:  # reuse finished predictions
                preds = load_preds(path)[0]
                sec = float(meta["s_per_frame"])
            else:
                preds, sec = predict_frames(
                    w, frames, use_sahi=sahi, device=device or _default_device()
                )
                save_preds(path, preds, {"weights": w.name, "sahi": sahi, "s_per_frame": sec})
            rows.append({"run": tag, "weights": w.name, "sahi": sahi, "s_per_frame": sec,
                         **score_run(preds, gts, conf)})  # fmt: skip
            typer.echo(f"{tag}: AP50={rows[-1]['ap50']:.3f} F1@{conf}={rows[-1]['f1']:.3f}")
    (out / "detector.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    typer.echo(f"-> {out / 'detector.json'}")


@eval_app.command("llm")
def eval_llm(
    preds: Annotated[Path, typer.Option(help="preds/<run>.json from `eval detector`")],
    conf: Annotated[float, typer.Option(help="Detections >= conf are sent to Claude")] = 0.25,
    model: Annotated[str, typer.Option(help="Claude model")] = "claude-opus-5-5",
    concurrency: int = 6,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the cost confirmation")] = False,
) -> None:
    """Verify every candidate with Claude and compare YOLO vs YOLO + Claude."""
    from rescuelens.data.heridal import ground_truth
    from rescuelens.eval.detection import load_preds
    from rescuelens.eval.llm_eval import (
        ablation,
        candidates,
        estimate_cost,
        save_verdicts,
        summarize,
        verify_candidates,
    )

    s = get_settings()
    all_preds, meta = load_preds(preds)
    cands = candidates(all_preds, conf)
    n = sum(len(v) for v in cands.values())
    est = estimate_cost(n, model)
    typer.echo(f"{n} candidates >= {conf} on {len(cands)} frames; estimated ~${est:.2f} ({model})")
    if est > s.llm_max_cost_usd:
        typer.echo(f"warning: above the ${s.llm_max_cost_usd:.2f} cap; the run will stop there")
    if not yes and not typer.confirm("Run the verification?"):
        raise typer.Abort()

    verdicts, tracker = asyncio.run(
        verify_candidates(
            cands, _test_dir() / "test_full" / "images", s, model=model, concurrency=concurrency
        )
    )
    result = {
        "preds": preds.name,
        "detector": meta,
        "conf": conf,
        "model": model,
        "summary": summarize(verdicts, tracker),
        "metrics": ablation(verdicts, ground_truth(_test_dir()), conf),
    }
    out = preds.parent.parent
    save_verdicts(out / "verdicts" / f"{preds.stem}_c{conf:g}_{model}.json", verdicts, result)
    (out / f"llm_{preds.stem}_c{conf:g}_{model}.json").write_text(json.dumps(result, indent=1))
    typer.echo(json.dumps({"summary": result["summary"], "metrics": result["metrics"]}, indent=1))


@eval_app.command("report")
def eval_report(out: Annotated[Path | None, typer.Option(help="Results directory")] = None) -> None:
    """Write README.md and pr_curve.png from the JSON results."""
    from rescuelens.eval.report import write_report

    typer.echo(f"-> {write_report(_results_dir(out), _test_dir())}")


if __name__ == "__main__":
    app()
