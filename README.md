# RescueLens

> Find missing people in drone imagery: YOLO + SAHI small-object detection, verified by Claude vision, shown on an operations map.

**Status:** work in progress (end-to-end pipeline working; training and evaluation next).

## Quickstart (dev)

```bash
uv tool install rust-just   # task runner (works on Windows/macOS/Linux)
just setup                  # auto-selects CUDA torch if an NVIDIA GPU is present
just check                  # lint + types + tests
just download-data          # HERIDAL, 8.3 GB (see DATASET.md)
just prepare-data           # 640 px YOLO tiles + full-resolution test frames
uv run rescuelens train     # fine-tune YOLO26 on the tiles (GPU recommended)
uv run rescuelens analyze path/to/drone/images   # writes runs/<run_id>/run.json
just run                    # backend on :8731
just ui                     # frontend on :5179
```

### How RescueLens reaches Claude

No API key is required if [Claude Code](https://claude.com/claude-code) is installed: the
`claude-code` backend runs `claude -p` headless (no tools, no MCP servers, no user settings)
with your Claude subscription, and validates the answer against the same JSON schema.
With `ANTHROPIC_API_KEY` set, the `api` backend calls the Messages API through the official
SDK instead. Select explicitly with `RESCUELENS_LLM_BACKEND=claude-code|api` (see `.env.example`).
Costs are reported as API list-price equivalents in both cases.

## License

Code: [AGPL-3.0](LICENSE) (required by the Ultralytics YOLO dependency).
Dataset licensing is documented separately in `DATASET.md`.
