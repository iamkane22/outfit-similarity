"""``outfit-search ingest | embed | evaluate | serve``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from outfit_search.config import DATASET_REPO, ENCODERS, default_paths

if TYPE_CHECKING:
    from outfit_search.encoders import OpenClipEncoder

app = typer.Typer(add_completion=False, help="Visual search for fashion catalogues.")
logger = logging.getLogger("outfit_search")

EncoderOption = Annotated[str, typer.Option(help=f"One of: {', '.join(ENCODERS)}")]


def _encoder(name: str) -> OpenClipEncoder:
    from outfit_search.encoders import OpenClipEncoder

    if name not in ENCODERS:
        raise typer.BadParameter(f"choose from {', '.join(ENCODERS)}")
    return OpenClipEncoder(name, ENCODERS[name])


@app.callback()
def _configure(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def ingest(
    sample: Annotated[
        int, typer.Option(help="Stratified sample size for the index; 0 keeps all 44k items.")
    ] = 12_000,
) -> None:
    """Download the catalogue from the Hugging Face Hub and unpack images."""
    from huggingface_hub import snapshot_download

    from outfit_search.catalogue import ingest as run_ingest

    local = Path(
        snapshot_download(DATASET_REPO, repo_type="dataset", allow_patterns=["data/*.parquet"])
    )
    paths = default_paths()
    items = run_ingest(
        sorted(local.glob("data/*.parquet")), paths.items, paths.images_dir, sample=sample
    )
    logger.info("%d items, %d article types", len(items), items["articleType"].nunique())


@app.command()
def embed(encoder: EncoderOption = "fashion-clip") -> None:
    """Embed every catalogue image and save the index."""
    from outfit_search.catalogue import iter_images, load_items
    from outfit_search.index import EmbeddingIndex

    paths = default_paths()
    items = load_items(paths.items)
    model = _encoder(encoder)
    vectors = model.encode_images(iter_images(items["id"], paths.images_dir))
    EmbeddingIndex(items["id"].to_numpy(), vectors, encoder).save(paths.index(encoder))
    logger.info("Saved %s (%d x %d)", paths.index(encoder), *vectors.shape)


@app.command()
def evaluate(
    encoder: EncoderOption = "fashion-clip",
    queries: Annotated[int, typer.Option(help="Random queries for search metrics.")] = 2000,
) -> None:
    """Score an index and write reports/metrics_<encoder>.json."""
    from outfit_search.catalogue import load_items
    from outfit_search.evaluation import evaluate as run_evaluation
    from outfit_search.index import EmbeddingIndex

    paths = default_paths()
    metrics = run_evaluation(
        EmbeddingIndex.load(paths.index(encoder)),
        _encoder(encoder),
        load_items(paths.items),
        queries,
    )
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (paths.reports_dir / f"metrics_{encoder}.json").write_text(json.dumps(metrics, indent=2))
    for key, value in metrics.items():
        typer.echo(f"{key:32s} {value:.4f}")


@app.command()
def report(
    examples: Annotated[
        str, typer.Option(help="Article types to show in the example grid, comma-separated.")
    ] = "Dresses,Sports Shoes,Handbags",
) -> None:
    """Compare clip and fashion-clip: table, significance and an example grid."""
    from outfit_search.catalogue import load_items
    from outfit_search.report import comparison_markdown, example_grid

    paths = default_paths()
    markdown = comparison_markdown(paths, "clip", "fashion-clip")
    (paths.reports_dir / "comparison.md").write_text(markdown)
    typer.echo(markdown)
    items = load_items(paths.items)
    query_ids = [
        int(items.loc[items["articleType"] == kind, "id"].iloc[0])
        for kind in examples.split(",")
        if (items["articleType"] == kind).any()
    ]
    example_grid(paths, ("clip", "fashion-clip"), query_ids)


@app.command()
def serve(
    encoder: EncoderOption = "fashion-clip",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start the HTTP API."""
    import os

    import uvicorn

    os.environ["OUTFIT_ENCODER"] = encoder
    uvicorn.run("outfit_search.api:app_from_env", factory=True, host=host, port=port)


if __name__ == "__main__":
    app()
