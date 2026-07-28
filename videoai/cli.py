"""VideoAI command line interface."""
from __future__ import annotations

from pathlib import Path

import typer

from videoai.config import load_config

app = typer.Typer(add_completion=False, help="Automated video pipeline.")


@app.callback()
def main() -> None:
    """Automated video pipeline."""


@app.command()
def config(path: Path = typer.Option(Path("config.yaml"), help="Config file path")) -> None:
    """Print the effective configuration."""
    typer.echo(load_config(path).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
