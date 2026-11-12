"""Command-line entry point: `uv run rescuelens --help`."""

import typer
import uvicorn

from rescuelens import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def version() -> None:
    """Print the RescueLens version."""
    typer.echo(__version__)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Start the FastAPI backend."""
    uvicorn.run("rescuelens.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
