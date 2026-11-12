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

COCO_WEIGHTS = PROJECT_ROOT / "weights" / "yolo26s.pt"


def _default_device() -> str:
    from rescuelens.detect.detector import auto_device

    return auto_device()


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
    weights: Annotated[
        Path | None, typer.Option(help="YOLO weights (default: HERIDAL fine-tune if present)")
    ] = None,
    device: Annotated[str | None, typer.Option(help="cpu | cuda:0 (default: auto)")] = None,
    sahi: Annotated[bool, typer.Option(help="Use SAHI slicing")] = True,
    conf: Annotated[float, typer.Option(help="Detector confidence threshold")] = 0.4,
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
        weights=weights or get_settings().default_weights,
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


@data_app.command("full")
def data_full(
    split: Annotated[str, typer.Option(help="train | val | test")] = "val",
) -> None:
    """Full-resolution frames + boxes.json for one split (val is used to pick thresholds)."""
    from rescuelens.data.heridal import export_full_split

    out = get_settings().data_dir / "heridal"
    n = export_full_split(out / "raw", out, split)
    typer.echo(f"{n} frames -> {out / f'{split}_full'}")


@data_app.command("samples")
def data_samples(
    prefix: Annotated[str, typer.Option(help="HERIDAL test series to copy")] = "test_BRA_",
    name: Annotated[str, typer.Option(help="Folder name under data/samples")] = "heridal-bra",
) -> None:
    """Copy one HERIDAL test flight (real EXIF GPS) to data/samples/<name> for the UI."""
    import shutil

    s = get_settings()
    src = s.data_dir / "heridal" / "test_full" / "images"
    boxes = json.loads((src.parent / "boxes.json").read_text(encoding="utf-8"))
    files = sorted(src.glob(f"{prefix}*.jpg"))
    if not files:
        raise typer.BadParameter(f"no {prefix}*.jpg in {src}; run `just prepare-data` first")
    dest = s.samples_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    ann: dict[str, list[list[float]]] = {}
    for f in files:
        shutil.copy2(f, dest / f.name)
        ann[f.name] = [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in boxes[f.stem]["boxes"]]
    (dest / "annotations.json").write_text(json.dumps(ann, indent=1), encoding="utf-8")
    typer.echo(f"{len(files)} frames (+ annotations.json) -> {dest}")


@app.command()
def train(
    data: Annotated[Path | None, typer.Option(help="dataset.yaml")] = None,
    weights: Annotated[Path, typer.Option(help="Starting checkpoint")] = COCO_WEIGHTS,
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


DEMO_ATTRIBUTION = """# Demo data

`run.json` is a recorded RescueLens analysis (`rescuelens export-demo`). The photos in
`frames/` and `crops/` are downscaled from the HERIDAL test set:

D. Božić-Štulić, Ž. Marušić, S. Gotovac, "Deep Learning Approach in Aerial Imagery for
Supporting Land Search and Rescue Missions", IJCV 127, 2019. Data: Zenodo record 5662351,
licensed CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/). Changes: resized and
cropped; detection boxes and Claude's verdicts added by RescueLens.
"""


@app.command("export-demo")
def export_demo(
    run_id: Annotated[str, typer.Argument(help="Finished run under runs/")],
    out: Annotated[Path, typer.Option(help="Static demo folder")] = (
        PROJECT_ROOT / "frontend" / "public" / "demo"
    ),
    max_side: Annotated[int, typer.Option(help="Longest side of the frame previews")] = 1400,
) -> None:
    """Copy a finished run (run.json + images) into the static replay demo."""
    import shutil

    from PIL import Image

    from rescuelens.schemas import Run

    src = get_settings().runs_dir / run_id
    run = Run.model_validate_json((src / "run.json").read_text(encoding="utf-8"))
    if run.status != "done":
        raise typer.BadParameter(f"run {run_id} is {run.status}, not done")
    if out.exists():
        shutil.rmtree(out)
    (out / "frames").mkdir(parents=True)
    shutil.copytree(src / "crops", out / "crops")
    for f in run.frames:
        with Image.open(src / f.preview) as im:
            im.thumbnail((max_side, max_side))
            im.save(out / f.preview, format="JPEG", quality=80, optimize=True)
    (out / "run.json").write_text(run.model_dump_json(), encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(DEMO_ATTRIBUTION, encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    typer.echo(
        f"{len(run.frames)} frames, {len(run.candidates)} candidates -> {out} ({size:.1f} MB)"
    )


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


bench_app = typer.Typer(
    no_args_is_help=True, help="Person-level benchmark: thresholds on val, test used once."
)
app.add_typer(bench_app, name="bench")


def _split_areas(frames: Path, agl_m: float) -> dict[str, float]:
    from rescuelens.ingest.camera import read_camera

    return {p.stem: read_camera(p).footprint_m2(agl_m) / 1e6 for p in sorted(frames.glob("*.jpg"))}


@bench_app.command("run")
def bench_run(
    weights: Annotated[Path | None, typer.Option(help="Checkpoint (default: fine-tuned)")] = None,
    sahi: Annotated[bool, typer.Option(help="Sliced inference")] = True,
    tag: Annotated[str | None, typer.Option(help="Name of this configuration")] = None,
    agl: Annotated[float, typer.Option(help="Assumed height above ground, m")] = 50.0,
    target: Annotated[list[float] | None, typer.Option(help="Recall targets (found)")] = None,
    claude: Annotated[
        list[Path] | None, typer.Option(help="Verdict files (test) for YOLO + Claude arms")
    ] = None,
    device: Annotated[str | None, typer.Option(help="cpu | cuda:0")] = None,
    boot: Annotated[int, typer.Option(help="Bootstrap replicates")] = 2000,
    out: Annotated[Path | None, typer.Option(help="Results directory")] = None,
) -> None:
    """Predict val + test (cached), choose thresholds on val, report test once."""
    from rescuelens.data.heridal import ground_truth
    from rescuelens.eval.bench_run import (
        Split,
        load_claude_arm,
        plot_curves,
        run_protocol,
        to_markdown,
    )
    from rescuelens.eval.detection import load_preds, predict_frames_timed, save_preds

    s = get_settings()
    weights = weights or s.default_weights
    tag = tag or f"{weights.stem}_{'sahi' if sahi else 'full'}"
    out = out or s.results_dir / "bench"
    root = _test_dir()
    splits: dict[str, Split] = {}
    for name in ("val", "test"):
        frames = root / f"{name}_full" / "images"
        if not frames.exists():
            raise typer.BadParameter(f"{frames} missing; run `rescuelens data full --split {name}`")
        path = out / "preds" / f"{tag}_{name}.json"
        if path.exists():
            preds, meta = load_preds(path)
        else:
            preds, timings = predict_frames_timed(
                weights, frames, use_sahi=sahi, device=device or _default_device()
            )
            meta = {"weights": weights.name, "sahi": sahi, "split": name, "timings": timings}
            save_preds(path, preds, meta)
        splits[name] = Split(
            name,
            preds,
            ground_truth(root, name),
            meta.get("timings", {}),
            _split_areas(frames, agl),
        )
    arms = [load_claude_arm(p, p.stem.rsplit("_", 1)[-1]) for p in claude or []]
    res = run_protocol(
        splits["val"],
        splits["test"],
        targets=target or [0.90, 0.95, 0.98],
        claude=arms,
        n_boot=boot,
    )
    res = {"tag": tag, "weights": weights.name, "sahi": sahi, "assumed_agl_m": agl, **res}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = to_markdown(res, f"Benchmark · {tag}")
    (out / f"{tag}.md").write_text(md, encoding="utf-8")
    plot_curves(res, out / f"{tag}_recall_fp.png")
    typer.echo(f"-> {out / f'{tag}.md'} (+ .json, _recall_fp.png)")


@bench_app.command("score")
def bench_score(
    preds: Annotated[Path, typer.Option(help="Predictions (save_preds format)")],
    gt: Annotated[Path, typer.Option(help="boxes.json (prepare format)")],
    conf: float = 0.4,
    boot: int = 200,
    check: Annotated[Path | None, typer.Option(help="Fail if results differ from this")] = None,
    write: Annotated[Path | None, typer.Option(help="Write the results here")] = None,
) -> None:
    """Score one prediction file at one threshold. CI runs this on a small fixture."""
    from rescuelens.data.voc import Box
    from rescuelens.eval.bench import CRITERIA, label_split, operating_point, threshold_for_recall
    from rescuelens.eval.detection import load_preds

    p, _ = load_preds(preds)
    raw = json.loads(gt.read_text(encoding="utf-8"))
    gts = {k: [Box(*b) for b in v["boxes"]] for k, v in raw.items()}
    res: dict[str, object] = {}
    for c in CRITERIA:
        lab = label_split(p, gts, c)
        res[c] = operating_point(lab, conf, n_boot=boot)
        res[f"{c}_conf_for_recall_0.9"] = threshold_for_recall(lab, 0.9)
    text = json.dumps(res, indent=1, sort_keys=True)
    if write:
        write.write_text(text + "\n", encoding="utf-8")
    typer.echo(text)
    if check:
        expected = json.loads(check.read_text(encoding="utf-8"))
        if not _close(expected, json.loads(text)):
            typer.echo(f"MISMATCH against {check}", err=True)
            raise typer.Exit(1)
        typer.echo(f"OK, matches {check}")


def _close(a: object, b: object, tol: float = 1e-9) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, int | float) and isinstance(b, int | float) and abs(a - b) <= tol
    return a == b


if __name__ == "__main__":
    app()
