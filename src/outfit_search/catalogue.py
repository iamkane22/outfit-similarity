"""Catalogue ingestion: Hugging Face parquet shards -> item table + image files.

The source dataset stores each product photo as encoded bytes inside parquet. We stream
the shards batch by batch (the whole set does not need to fit in memory), write every
image to ``images/<id>.jpg`` for cheap random access, and keep the metadata in a small
parquet table.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

logger = logging.getLogger(__name__)

METADATA_COLUMNS: tuple[str, ...] = (
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "usage",
    "productDisplayName",
)


def stratified_sample(items: pd.DataFrame, size: int, seed: int = 0) -> pd.DataFrame:
    """About ``size`` items, every article type keeping its share (and at least one item).

    Used to keep the catalogue small enough to embed quickly on a laptop while every
    category stays represented, so retrieval metrics are not skewed towards the big ones.
    """
    if size <= 0 or size >= len(items):
        return items.reset_index(drop=True)
    fraction = size / len(items)
    parts = [
        group.sample(max(1, round(len(group) * fraction)), random_state=seed)
        for _, group in items.groupby("articleType")
    ]
    return pd.concat(parts).sort_values("id").reset_index(drop=True)


def ingest(
    shards: Sequence[Path],
    items_path: Path,
    images_dir: Path,
    *,
    batch_size: int = 512,
    sample: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Write images and the item table; returns the item table.

    With ``sample > 0`` the item table keeps a stratified sample of that size.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=batch_size):
            for record in batch.to_pylist():
                image = record.get("image") or {}
                payload = image.get("bytes")
                if not payload or not record.get("articleType"):
                    continue
                target = images_dir / f"{record['id']}.jpg"
                if not target.exists():
                    Image.open(io.BytesIO(payload)).convert("RGB").save(target, quality=92)
                rows.append({col: record.get(col) for col in METADATA_COLUMNS})
        logger.info("Ingested %s (%d items so far)", shard.name, len(rows))

    items = pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)
    items["id"] = items["id"].astype("int64")
    items = stratified_sample(items, sample, seed)
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(items_path, index=False)
    return items


def load_items(items_path: Path) -> pd.DataFrame:
    if not items_path.exists():
        raise FileNotFoundError(f"{items_path} not found; run `outfit-search ingest` first")
    return pd.read_parquet(items_path)


def iter_images(ids: Iterable[int], images_dir: Path) -> Iterator[Image.Image]:
    for item_id in ids:
        with Image.open(images_dir / f"{item_id}.jpg") as img:
            yield img.convert("RGB")
