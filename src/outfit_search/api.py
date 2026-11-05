"""HTTP API: text search, image search and "more like this".

    uvicorn --factory outfit_search.api:app_from_env

All endpoints accept optional ``gender`` and ``category`` (masterCategory) filters, which
are applied inside the index so ``k`` results are always returned when they exist.
"""

from __future__ import annotations

import io
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from outfit_search.encoders import Embeddings, Encoder
from outfit_search.index import EmbeddingIndex


class Hit(BaseModel):
    id: int
    name: str
    article_type: str
    colour: str | None
    score: float


class SearchResponse(BaseModel):
    encoder: str
    hits: list[Hit]


class TextQuery(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    k: int = Field(default=12, ge=1, le=100)
    gender: str | None = None
    category: str | None = None


def create_app(index: EmbeddingIndex, items: pd.DataFrame, encoder: Encoder) -> FastAPI:
    if not np.array_equal(items["id"].to_numpy(), index.ids):
        raise ValueError("items and index are not aligned")
    app = FastAPI(title="Outfit similarity search", version="0.1.0")

    def candidates(gender: str | None, category: str | None) -> np.ndarray | None:
        mask = np.ones(len(items), dtype=bool)
        if gender:
            mask &= (items["gender"] == gender).to_numpy()
        if category:
            mask &= (items["masterCategory"] == category).to_numpy()
        return None if mask.all() else mask

    def respond(
        query: Embeddings, k: int, mask: np.ndarray | None, exclude: int | None = None
    ) -> SearchResponse:
        excluded = None if exclude is None else np.array([exclude])
        result = index.search(query, k, exclude=excluded, candidates=mask)
        hits = []
        for pos, score in zip(result.positions[0], result.scores[0], strict=True):
            if not np.isfinite(score):  # fewer candidates than k
                continue
            row = items.iloc[int(pos)]
            hits.append(
                Hit(
                    id=int(row["id"]),
                    name=str(row["productDisplayName"]),
                    article_type=str(row["articleType"]),
                    colour=None if pd.isna(row["baseColour"]) else str(row["baseColour"]),
                    score=float(score),
                )
            )
        return SearchResponse(encoder=index.encoder, hits=hits)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "encoder": index.encoder, "items": len(index)}

    @app.post("/search/text")
    def search_text(body: TextQuery) -> SearchResponse:
        return respond(
            encoder.encode_texts([body.query]), body.k, candidates(body.gender, body.category)
        )

    @app.post("/search/image")
    def search_image(
        file: Annotated[UploadFile, File(description="JPEG or PNG photo")],
        k: Annotated[int, Query(ge=1, le=100)] = 12,
        gender: str | None = None,
        category: str | None = None,
    ) -> SearchResponse:
        try:
            image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        except UnidentifiedImageError as error:
            raise HTTPException(status_code=415, detail="not an image") from error
        return respond(encoder.encode_images([image]), k, candidates(gender, category))

    @app.get("/items/{item_id}/similar")
    def similar(
        item_id: int,
        k: Annotated[int, Query(ge=1, le=100)] = 12,
        gender: str | None = None,
        category: str | None = None,
    ) -> SearchResponse:
        try:
            position = index.position(item_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"unknown item {item_id}") from error
        return respond(index.vectors[position], k, candidates(gender, category), exclude=position)

    return app


def app_from_env() -> FastAPI:
    """Factory for uvicorn: loads the index named by ``OUTFIT_ENCODER`` (default fashion-clip)."""
    import os

    from outfit_search.catalogue import load_items
    from outfit_search.config import ENCODERS, default_paths
    from outfit_search.encoders import OpenClipEncoder

    name = os.environ.get("OUTFIT_ENCODER", "fashion-clip")
    paths = default_paths()
    return create_app(
        EmbeddingIndex.load(paths.index(name)),
        load_items(paths.items),
        OpenClipEncoder(name, ENCODERS[name]),
    )
