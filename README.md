# RescueLens

> Find missing people in drone imagery: YOLO + SAHI small-object detection, verified by Claude vision, shown on an operations map.

**Status:** work in progress (phase 0: project scaffold).

## Quickstart (dev)

```bash
uv tool install rust-just   # task runner (works on Windows/macOS/Linux)
just setup                  # auto-selects CUDA torch if an NVIDIA GPU is present
cp .env.example .env        # add your ANTHROPIC_API_KEY
just check                  # lint + types + tests
just run                    # backend on :8731
just ui                     # frontend on :5179
```

## License

Code: [AGPL-3.0](LICENSE) (required by the Ultralytics YOLO dependency).
Dataset licensing is documented separately in `DATASET.md`.
