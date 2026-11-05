"""A tiny synthetic catalogue and a deterministic encoder, so tests need no model or GPU."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from outfit_search.encoders import Embeddings, normalise
from outfit_search.index import EmbeddingIndex

COLOURS = {"Navy Blue": (20, 30, 110), "Red": (200, 20, 30), "White": (245, 245, 245)}
WORDS = {"navy": "Navy Blue", "red": "Red", "white": "White"}


class FakeEncoder:
    """Embeds images by mean colour and texts by the colour word they contain."""

    name = "fake"

    def encode_images(self, images: Iterable[Image.Image]) -> Embeddings:
        rows = [
            np.asarray(img.convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
            for img in images
        ]
        return normalise(np.array(rows) / 255.0 + 0.01)

    def encode_texts(self, texts: Sequence[str]) -> Embeddings:
        rows = []
        for text in texts:
            colour = next(
                (COLOURS[name] for word, name in WORDS.items() if word in text.lower()),
                (128, 128, 128),
            )
            rows.append(np.asarray(colour, dtype=float) / 255.0 + 0.01)
        return normalise(np.array(rows))


@pytest.fixture
def items() -> pd.DataFrame:
    rows = []
    item_id = 100
    for article in ("Tshirts", "Jeans"):
        for colour in COLOURS:
            for gender in ("Men", "Women"):
                rows.append(
                    {
                        "id": item_id,
                        "gender": gender,
                        "masterCategory": "Apparel",
                        "subCategory": "Topwear" if article == "Tshirts" else "Bottomwear",
                        "articleType": article,
                        "baseColour": colour,
                        "season": "Summer",
                        "usage": "Casual",
                        "productDisplayName": f"{gender} {colour} {article}",
                    }
                )
                item_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def images_dir(tmp_path: Path, items: pd.DataFrame) -> Path:
    target = tmp_path / "images"
    target.mkdir()
    rng = np.random.default_rng(0)
    for row in items.itertuples():
        base = np.array(COLOURS[row.baseColour], dtype=float)
        pixels = np.clip(base + rng.normal(0, 4, (16, 16, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(target / f"{row.id}.jpg")
    return target


@pytest.fixture
def index(items: pd.DataFrame, images_dir: Path) -> EmbeddingIndex:
    from outfit_search.catalogue import iter_images

    vectors = FakeEncoder().encode_images(iter_images(items["id"], images_dir))
    return EmbeddingIndex(items["id"].to_numpy(), vectors, "fake")
