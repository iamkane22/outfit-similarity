from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from outfit_search.catalogue import METADATA_COLUMNS, ingest, load_items


def _jpeg(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), colour, dtype=np.uint8)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _shard(path: Path) -> Path:
    records = [
        {"id": 1, "articleType": "Tshirts", "image": {"bytes": _jpeg((200, 0, 0)), "path": None}},
        {"id": 2, "articleType": None, "image": {"bytes": _jpeg((0, 200, 0)), "path": None}},
        {"id": 3, "articleType": "Jeans", "image": {"bytes": None, "path": None}},
        {"id": 4, "articleType": "Jeans", "image": {"bytes": _jpeg((0, 0, 200)), "path": None}},
    ]
    for record in records:
        for column in METADATA_COLUMNS:
            record.setdefault(column, "x")
    pq.write_table(pa.Table.from_pylist(records), path)
    return path


def test_ingest_writes_images_and_skips_unusable_rows(tmp_path: Path) -> None:
    shard = _shard(tmp_path / "train-0.parquet")
    items = ingest([shard], tmp_path / "items.parquet", tmp_path / "images", batch_size=2)

    assert items["id"].tolist() == [1, 4]
    assert sorted(p.name for p in (tmp_path / "images").iterdir()) == ["1.jpg", "4.jpg"]
    assert list(load_items(tmp_path / "items.parquet").columns) == list(METADATA_COLUMNS)
    with Image.open(tmp_path / "images" / "4.jpg") as img:
        assert img.size == (8, 8)


def test_stratified_sample_keeps_every_article_type() -> None:
    import pandas as pd

    from outfit_search.catalogue import stratified_sample

    items = pd.DataFrame(
        {"id": range(1000), "articleType": ["Tshirts"] * 900 + ["Ties"] * 95 + ["Rare"] * 5}
    )
    sample = stratified_sample(items, 100)
    counts = sample["articleType"].value_counts()
    assert counts["Tshirts"] == 90
    assert counts["Ties"] in (9, 10)
    assert counts["Rare"] == 1
    assert sample["id"].is_monotonic_increasing
    assert len(stratified_sample(items, 0)) == 1000
