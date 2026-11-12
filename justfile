# RescueLens task runner. Install `just` with: uv tool install rust-just

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Torch build: cu128 if an NVIDIA GPU is visible, else cpu. Override with RESCUELENS_TORCH.
torch := env("RESCUELENS_TORCH", `uv run --no-project python -c "import shutil; print('cu128' if shutil.which('nvidia-smi') else 'cpu')"`)
uvr := "uv run --extra " + torch

# List available recipes
default:
    @just --list

# Install Python + frontend deps (auto-selects CUDA torch when an NVIDIA GPU is present)
setup:
    uv sync --extra {{torch}}
    cd frontend; npm ci

# Lint, type-check and test everything
check: lint typecheck test

lint:
    {{uvr}} ruff check .
    {{uvr}} ruff format --check .
    cd frontend; npm run lint
    cd frontend; npm run format:check

# Auto-fix formatting
fmt:
    {{uvr}} ruff check --fix .
    {{uvr}} ruff format .
    cd frontend; npm run format

typecheck:
    {{uvr}} mypy
    cd frontend; npm run typecheck

test:
    {{uvr}} pytest -q
    cd frontend; npm test

# Download HERIDAL (8.3 GB from Zenodo, resumable, md5-verified)
download-data:
    {{uvr}} rescuelens data download

# Extract HERIDAL and build 640 px YOLO tiles + full-resolution test frames
prepare-data:
    {{uvr}} rescuelens data prepare

# Copy one real HERIDAL flight (12 frames, EXIF GPS) to data/samples/heridal-bra
samples:
    {{uvr}} rescuelens data samples

# Run backend (port 8731) and frontend dev server (port 5179)
run:
    {{uvr}} rescuelens serve

ui:
    cd frontend; npm run dev


# Export a finished run and build the static replay demo into frontend/dist
demo run_id:
    {{uvr}} rescuelens export-demo {{run_id}}
    cd frontend; $env:VITE_DEMO = '1'; npm run build
