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

# Download HERIDAL (8.3 GB from Zenodo, resumable, md5-verified)
download-data:
    {{uvr}} rescuelens data download

# Extract HERIDAL and build 640 px YOLO tiles + full-resolution test frames
prepare-data:
    {{uvr}} rescuelens data prepare

# Run backend (port 8731) and frontend dev server (port 5179)
run:
    {{uvr}} rescuelens serve

ui:
    cd frontend; npm run dev

