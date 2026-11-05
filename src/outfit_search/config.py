"""Paths and the encoder registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATASET_REPO = "ashraq/fashion-product-images-small"

#: Short name -> open_clip model identifier on the Hugging Face Hub.
ENCODERS: dict[str, str] = {
    "clip": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "fashion-clip": "hf-hub:Marqo/marqo-fashionCLIP",
}


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    reports_dir: Path

    @property
    def items(self) -> Path:
        return self.data_dir / "catalogue" / "items.parquet"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "catalogue" / "images"

    def index(self, encoder: str) -> Path:
        return self.data_dir / "indexes" / f"{encoder}.npz"

    def image(self, item_id: int) -> Path:
        return self.images_dir / f"{item_id}.jpg"


def default_paths() -> Paths:
    return Paths(
        data_dir=Path(os.environ.get("OUTFIT_DATA_DIR", "data")),
        reports_dir=Path(os.environ.get("OUTFIT_REPORTS_DIR", "reports")),
    )
