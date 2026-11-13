"""Command-line entry point: `uv run rescuelens --help`."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from pathlib import Path
from typing import Annotated, Any

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


@data_app.command("hardneg")
def data_hardneg(
    weights: Annotated[Path | None, typer.Option(help="Detector that mines (default)")] = None,
    slice_size: Annotated[int, typer.Option("--slice", help="SAHI tile size, px")] = 1024,
    tile_imgsz: Annotated[int, typer.Option(help="Network input size per tile")] = 1280,
    min_score: Annotated[float, typer.Option(help="Lowest false-alarm score mined")] = 0.10,
    max_per_photo: Annotated[int, typer.Option(help="Hard tiles per photo, at most")] = 8,
    device: Annotated[str | None, typer.Option(help="cpu | cuda:0")] = None,
) -> None:
    """Mine the detector's false alarms on the full *train* photos into extra training tiles.

    Writes data/heridal/tiles_hn/ (tiles, labels, dataset.yaml listing both tile trees).
    Detections on train are cached in train_full/preds_<weights>.json.
    """
    from rescuelens.data.hardneg import HardNegConfig, export_hard_negatives
    from rescuelens.data.heridal import export_full_split, ground_truth
    from rescuelens.eval.detection import load_preds, predict_frames_timed, save_preds

    s = get_settings()
    weights = weights or s.default_weights
    out = s.data_dir / "heridal"
    if not (out / "train_full" / "boxes.json").exists():
        export_full_split(out / "raw", out, "train")
    frames = out / "train_full" / "images"
    cache = out / "train_full" / f"preds_{weights.stem}_s{slice_size}_i{tile_imgsz}.json"
    detector = {"slice_size": slice_size, "tile_image_size": tile_imgsz}
    if cache.exists():
        preds, _ = load_preds(cache)
    else:
        preds, timings = predict_frames_timed(
            weights, frames, use_sahi=True, device=device or _default_device(), detector=detector
        )
        save_preds(cache, preds, {"weights": weights.name, "split": "train",
                                  "detector": detector, "timings": timings})  # fmt: skip
    cfg = HardNegConfig(min_score=min_score, max_per_photo=max_per_photo)
    summary = export_hard_negatives(preds, ground_truth(out, "train"), frames, out, cfg)
    typer.echo(json.dumps(summary, indent=1))


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
    scale: Annotated[float, typer.Option(help="Random zoom range: 1 +/- scale")] = 0.5,
    eval_split: Annotated[str, typer.Option(help="Split scored after training")] = "val",
    workers: Annotated[int, typer.Option(help="Data-loading processes (RAM grows with it)")] = 8,
    resume: Annotated[bool, typer.Option(help="Continue run NAME from its last.pt")] = False,
) -> None:
    """Fine-tune YOLO on HERIDAL tiles, then score the best checkpoint on val tiles."""
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
        scale=scale,
        eval_split=eval_split,
        workers=workers,
        resume=resume,
    )
    summary = train_detector(cfg)
    typer.echo(json.dumps(summary["tiles"], indent=1))


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
    slice_size: Annotated[
        int | None, typer.Option("--slice", help="SAHI tile size, px (default: detector's)")
    ] = None,
    tile_imgsz: Annotated[int | None, typer.Option(help="Network input size per tile")] = None,
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
            detector: dict[str, int] = {}
            if slice_size:
                detector["slice_size"] = slice_size
            if tile_imgsz:
                detector["tile_image_size"] = tile_imgsz
            preds, timings = predict_frames_timed(
                weights,
                frames,
                use_sahi=sahi,
                device=device or _default_device(),
                detector=detector,
            )
            meta = {"weights": weights.name, "sahi": sahi, "split": name,
                    "detector": detector, "timings": timings}  # fmt: skip
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


@bench_app.command("strata")
def bench_strata(
    tag: Annotated[str, typer.Option(help="Configuration (cached val predictions)")] = (
        "rescuelens-yolo26s-heridal_sahi"
    ),
    conf: float = 0.4,
    out: Annotated[Path | None, typer.Option(help="Results directory")] = None,
) -> None:
    """Recall on val by person size, ground resolution and camera.

    Test is only *described* (annotation sizes, EXIF), never scored here: val's recall per
    size bin, weighted by a test group's sizes, is what scale alone would predict.
    """
    import numpy as np

    from rescuelens.data.heridal import ground_truth
    from rescuelens.eval.bench import CRITERIA, label_split
    from rescuelens.eval.detection import SCORE_FLOOR, load_preds
    from rescuelens.eval.strata import (
        GSD_BINS_CM,
        SIZE_BINS_PX,
        PhotoMeta,
        binner,
        found_flags,
        people,
        recall_by,
        strata_markdown,
    )

    out = out or get_settings().results_dir / "bench"
    root = _test_dir()
    size_bin = binner(SIZE_BINS_PX, "px")
    gsd_bin = binner(GSD_BINS_CM, "cm/px")

    gts = ground_truth(root, "val")
    meta = {s: PhotoMeta.read(root / "val_full" / "images" / f"{s}.jpg") for s in gts}
    preds, _ = load_preds(out / "preds" / f"{tag}_val.json")
    tables: dict[str, list[dict[str, object]]] = {}
    for crit in CRITERIA:
        lab = label_split(preds, gts, crit)
        ps = people(lab, gts, meta)
        flags = {
            f"conf {conf:.2f}": found_flags(lab, gts, conf),
            "any box (conf 0.05)": found_flags(lab, gts, SCORE_FLOOR),
        }
        n = lab.n_images
        tables[f"{crit}/size"] = recall_by(ps, flags, lambda p: size_bin(p.size_px), n_images=n)
        if crit == "found":
            tables["found/gsd"] = recall_by(ps, flags, lambda p: gsd_bin(p.gsd_cm), n_images=n)
            tables["found/camera"] = recall_by(ps, flags, lambda p: p.camera, n_images=n)

    # test: annotations and EXIF only
    tgts = ground_truth(root, "test")
    tmeta = {s: PhotoMeta.read(root / "test_full" / "images" / f"{s}.jpg") for s in tgts}
    val_by_size = {
        r["stratum"]: r[f"conf {conf:.2f}"]["recall"]  # type: ignore[index]
        for r in tables["found/size"]
    }
    groups: dict[str, list[float]] = {}
    alts: dict[str, list[float]] = {}
    for stem, boxes in tgts.items():
        cam = tmeta[stem].camera
        groups.setdefault(cam, []).extend(math.sqrt(b.w * b.h) for b in boxes)
        if (alt := tmeta[stem].rel_alt_m) is not None:
            alts.setdefault(cam, []).append(alt)
    described = []
    for cam, sizes in sorted(groups.items()):
        bins = [size_bin(x) for x in sizes]
        expected = sum(val_by_size.get(b, float("nan")) for b in bins if b) / max(len(bins), 1)
        described.append(
            {
                "camera": cam,
                "people": len(sizes),
                "size_px_median": float(np.median(sizes)) if sizes else None,
                "share_below_40px": float(np.mean([x < 40 for x in sizes])) if sizes else None,
                "rel_alt_m_median": float(np.median(alts[cam])) if cam in alts else None,
                "recall_expected_from_size": expected,
            }
        )
    res = {"tag": tag, "conf": conf, "val": tables, "test_described": described}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"strata_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / f"strata_{tag}.md").write_text(strata_markdown(res), encoding="utf-8")
    typer.echo(f"-> {out / f'strata_{tag}.md'}")


@bench_app.command("val")
def bench_val(
    tag: Annotated[str, typer.Option(help="Name of this configuration")],
    weights: Annotated[Path | None, typer.Option(help="Checkpoint (default: fine-tuned)")] = None,
    slice_size: Annotated[int, typer.Option("--slice", help="SAHI tile size, px")] = 640,
    tile_imgsz: Annotated[int, typer.Option(help="Network input size per tile")] = 1280,
    overlap: float = 0.2,
    standard_pred: Annotated[bool, typer.Option(help="Extra whole-frame pass")] = True,
    hflip: Annotated[
        bool, typer.Option(help="Detect on the mirrored photo (one TTA view)")
    ] = False,
    gsd_target: Annotated[
        float | None,
        typer.Option(help="Rescale photos with DJI altitude to this cm/px (others unchanged)"),
    ] = None,
    order: Annotated[int, typer.Option(help="Row order in the table")] = 0,
    device: Annotated[str | None, typer.Option(help="cpu | cuda:0")] = None,
    out: Annotated[Path | None, typer.Option(help="Results directory")] = None,
) -> None:
    """Ablation on val only: predict (cached), score at false-alarm budgets, update the table."""
    from rescuelens.eval.detection import load_preds, predict_frames_timed, save_preds
    from rescuelens.eval.strata import PhotoMeta

    s = get_settings()
    weights = weights or s.default_weights
    out = out or s.results_dir / "bench"
    frames = _test_dir() / "val_full" / "images"
    detector = {
        "slice_size": slice_size,
        "tile_image_size": tile_imgsz,
        "overlap": overlap,
        "standard_pred": standard_pred,
        "hflip": hflip,
    }

    def slice_for(p: Path) -> int | None:
        gsd = PhotoMeta.read(p).gsd_cm
        if gsd_target is None or gsd is None:
            return None
        # rescale by gsd / target: a smaller tile, resized to tile_imgsz, magnifies
        return max(256, min(2048, round(slice_size * gsd_target / gsd)))

    path = out / "preds" / f"{tag}_val.json"
    if path.exists():
        preds, meta = load_preds(path)
    else:
        preds, timings = predict_frames_timed(
            weights,
            frames,
            use_sahi=True,
            device=device or _default_device(),
            detector=detector,
            slice_for=slice_for if gsd_target else None,
        )
        meta = {"weights": weights.name, "split": "val", "detector": detector,
                "gsd_target": gsd_target, "timings": timings}  # fmt: skip
        save_preds(path, preds, meta)
    upscale = (
        f"{tile_imgsz / slice_size:.2f}x"
        + (f", GSD {gsd_target} cm/px" if gsd_target else "")
        + (", mirrored" if hflip else "")
    )
    row = {"tag": tag, "order": order, "upscale": upscale, "weights": weights.name,
           "detector": detector, "gsd_target": gsd_target}  # fmt: skip
    _ablation_row(out, row, preds, meta.get("timings", {}))


def _ablation_row(
    out: Path, row: dict[str, Any], preds: dict[str, Any], timings: dict[str, Any]
) -> None:
    """Score cached val predictions, save the row and rebuild the ablation table."""
    from rescuelens.data.heridal import ground_truth
    from rescuelens.eval.ablation import collect, score_val, table
    from rescuelens.eval.bench_run import Split

    row["score"] = score_val(Split("val", preds, ground_truth(_test_dir(), "val"), timings))
    (out / "ablation").mkdir(parents=True, exist_ok=True)
    path = out / "ablation" / f"{row['tag']}.json"
    path.write_text(json.dumps(row, indent=1), encoding="utf-8")
    (out / "ablation_val.md").write_text(table(collect(out / "ablation")), encoding="utf-8")
    b = row["score"]["budgets"]["found"]["1"]
    typer.echo(f"{row['tag']}: recall@<=1FA/photo (found) = {b and b['recall']}")


@bench_app.command("fuse")
def bench_fuse(
    tag: Annotated[str, typer.Option(help="Name of the fused configuration")],
    member: Annotated[list[str], typer.Option(help="Tags whose cached val preds are merged")],
    iou_thr: Annotated[float, typer.Option("--iou", help="Fusion IoU")] = 0.55,
    order: Annotated[int, typer.Option(help="Row order in the table")] = 0,
    out: Annotated[Path | None, typer.Option(help="Results directory")] = None,
) -> None:
    """Weighted box fusion of several cached val runs (flip TTA, model ensembles).

    Detect time per photo is the sum of the members' times: the views run one after another.
    """
    from rescuelens.data.voc import Box
    from rescuelens.detect.fusion import fuse_views
    from rescuelens.eval.detection import load_preds, save_preds
    from rescuelens.eval.metrics import Pred

    out = out or get_settings().results_dir / "bench"
    runs = [load_preds(out / "preds" / f"{m}_val.json") for m in member]
    stems = sorted(set().union(*(p.keys() for p, _ in runs)))
    preds: dict[str, list[Pred]] = {}
    timings: dict[str, dict[str, float]] = {}
    for st in stems:
        views = [
            [(q.box.xmin, q.box.ymin, q.box.xmax, q.box.ymax, q.score) for q in p.get(st, [])]
            for p, _ in runs
        ]
        preds[st] = [Pred(Box(*f[:4]), f[4]) for f in fuse_views(views, iou_thr)]
        ts = [m.get("timings", {}).get(st) for _, m in runs]
        if all(ts):
            timings[st] = {k: sum(t[k] for t in ts if t) for k in ("load_s", "detect_s")}
    meta = {"split": "val", "fused": member, "iou": iou_thr, "timings": timings}
    save_preds(out / "preds" / f"{tag}_val.json", preds, meta)
    weights = sorted({str(m.get("weights")) for _, m in runs})
    row = {"tag": tag, "order": order, "upscale": f"WBF of {len(member)} runs",
           "weights": ", ".join(weights), "fused": member, "iou": iou_thr}  # fmt: skip
    _ablation_row(out, row, preds, timings)


@bench_app.command("compare")
def bench_compare(
    pair: Annotated[list[str], typer.Option(help="BASE:CANDIDATE tags (cached val preds)")],
    budget: Annotated[float, typer.Option(help="False alarms per photo allowed")] = 1.0,
    out: Annotated[Path | None, typer.Option(help="Results directory")] = None,
) -> None:
    """Paired bootstrap of recall differences on val, each config at its own budget threshold."""
    from rescuelens.data.heridal import ground_truth
    from rescuelens.eval.bench import (
        CRITERIA,
        label_split,
        paired_difference,
        threshold_for_fp_budget,
    )
    from rescuelens.eval.detection import load_preds

    out = out or get_settings().results_dir / "bench"
    gts = ground_truth(_test_dir(), "val")
    lines = [
        f"# Paired comparisons · val · recall at ≤ {budget:g} false alarms per photo",
        "",
        "Candidate minus base, same photos in every bootstrap replicate (2,000).",
        "",
        "| Base | Candidate | Criterion | Δ recall (points) [95 % CI] | "
        "Replicates where candidate wins |",
        "|---|---|---|---:|---:|",
    ]
    for item in pair:
        a_tag, b_tag = item.split(":")
        pa, _ = load_preds(out / "preds" / f"{a_tag}_val.json")
        pb, _ = load_preds(out / "preds" / f"{b_tag}_val.json")
        for c in CRITERIA:
            la, lb = label_split(pa, gts, c), label_split(pb, gts, c)
            ta, tb = threshold_for_fp_budget(la, budget), threshold_for_fp_budget(lb, budget)
            if ta is None or tb is None:
                continue
            d = paired_difference(la, ta, lb, tb)
            lo, hi = d["ci"]
            crit = "found" if c == "found" else "IoU ≥ 0.5"
            lines.append(
                f"| {a_tag} | {b_tag} | {crit} | {100 * d['difference']:+.1f} "
                f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {100 * d['share_above_zero']:.0f} % |"
            )
    path = out / f"ablation_pairs_fa{budget:g}.md"
    # Keep earlier pairs: replace rows with the same (base, candidate, criterion), add the rest.
    head = lines[:6]

    def key(row: str) -> tuple[str, ...]:
        return tuple(c.strip() for c in row.split("|")[1:4])

    new = {key(r): r for r in lines[6:]}
    old = path.read_text(encoding="utf-8").splitlines()[6:] if path.exists() else []
    kept = [r for r in old if r.startswith("|") and key(r) not in new]
    path.write_text("\n".join(head + kept + list(new.values())) + "\n", encoding="utf-8")
    typer.echo(f"-> {path}")


@bench_app.command("latency")
def bench_latency(
    tag: Annotated[list[str], typer.Option(help="Ablation tags (results/bench/ablation/*.json)")],
    photos: Annotated[int, typer.Option(help="Val photos to time")] = 40,
    rounds: Annotated[int, typer.Option(help="Passes over the photos")] = 2,
    device: Annotated[str | None, typer.Option(help="cpu | cuda:0")] = None,
    out: Annotated[Path | None, typer.Option(help="Results directory")] = None,
) -> None:
    """Time configurations interleaved photo by photo, so drift (heat, other load) hits all."""
    import time

    import numpy as np
    from PIL import Image

    from rescuelens.detect.detector import DetectorConfig, PersonDetector
    from rescuelens.eval.detection import SCORE_FLOOR
    from rescuelens.eval.strata import PhotoMeta

    s = get_settings()
    out = out or s.results_dir / "bench"
    dev = device or _default_device()
    cfgs = {t: json.loads((out / "ablation" / f"{t}.json").read_text("utf-8")) for t in tag}
    dets = {
        t: PersonDetector(
            DetectorConfig(
                weights=PROJECT_ROOT / "weights" / c["weights"],
                confidence=SCORE_FLOOR,
                device=dev,
                **c["detector"],
            )
        )
        for t, c in cfgs.items()
    }
    paths = sorted((_test_dir() / "val_full" / "images").glob("*.jpg"))[:: max(1, 155 // photos)]
    paths = paths[:photos]
    arrays = []
    for p in paths:
        with Image.open(p) as im:
            arrays.append((p, np.asarray(im.convert("RGB"))))

    def tile_for(t: str, p: Path) -> int | None:
        target = cfgs[t]["gsd_target"] and float(cfgs[t]["gsd_target"])
        gsd = PhotoMeta.read(p).gsd_cm if target else None
        if not target or gsd is None:
            return None
        size = int(cfgs[t]["detector"]["slice_size"])
        return max(256, min(2048, round(size * target / gsd)))

    tiles = {(t, p): tile_for(t, p) for t in dets for p, _ in arrays}  # EXIF read untimed
    for t, d in dets.items():  # warm-up
        d.detect(arrays[0][1], slice_size=tiles[t, arrays[0][0]])
    times: dict[str, list[float]] = {t: [] for t in dets}
    for _ in range(rounds):
        for p, arr in arrays:
            for t, d in dets.items():
                t0 = time.perf_counter()
                d.detect(arr, slice_size=tiles[t, p])
                times[t].append(time.perf_counter() - t0)
    lines = [
        f"# Detector latency · {len(paths)} val photos, {rounds} rounds, interleaved",
        "",
        f"Device: {dev}. Detector only (tiling, inference, merging); JPEG decoding excluded.",
        "",
        "| Configuration | p50 | p95 | mean |",
        "|---|---:|---:|---:|",
    ]
    for t, v in times.items():
        a = np.asarray(v)
        p50, p95 = np.percentile(a, [50, 95])
        lines.append(f"| {t} | {p50:.2f} s | {p95:.2f} s | {a.mean():.2f} s |")
    path = out / "ablation_latency.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"-> {path}")


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
