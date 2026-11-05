"""Exact cosine-similarity index over normalised embeddings.

For catalogues up to a few hundred thousand items a brute-force matrix product is fast
(tens of milliseconds on a laptop), exact, and has no build step, so an approximate
index (FAISS, HNSW) would add complexity without a measurable benefit. Queries are
processed in chunks so memory stays bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from outfit_search.encoders import Embeddings


@dataclass(frozen=True)
class SearchResult:
    positions: NDArray[np.intp]  # (n_queries, k) row positions in the index
    scores: NDArray[np.float32]  # (n_queries, k) cosine similarities, descending


class EmbeddingIndex:
    def __init__(self, ids: NDArray[np.int64], vectors: Embeddings, encoder: str) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        norms = np.linalg.norm(vectors, axis=1)
        if len(vectors) and not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("vectors must be L2-normalised")
        self.ids = np.asarray(ids, dtype=np.int64)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.encoder = encoder
        self._position = {int(item_id): i for i, item_id in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def position(self, item_id: int) -> int:
        return self._position[int(item_id)]

    def search(
        self,
        queries: Embeddings,
        k: int = 10,
        *,
        exclude: NDArray[np.intp] | None = None,
        candidates: NDArray[np.bool_] | None = None,
        chunk_size: int = 1024,
    ) -> SearchResult:
        """Top-``k`` neighbours of each query.

        ``exclude`` holds, per query, one index position to drop (the query item itself
        in item-to-item search). ``candidates`` restricts results to a boolean mask over
        the index, used for filters such as gender or category.
        """
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        k = min(k, len(self))
        positions = np.empty((len(queries), k), dtype=np.intp)
        scores = np.empty((len(queries), k), dtype=np.float32)
        for start in range(0, len(queries), chunk_size):
            block = queries[start : start + chunk_size] @ self.vectors.T
            if candidates is not None:
                block[:, ~candidates] = -np.inf
            if exclude is not None:
                rows = np.arange(len(block))
                block[rows, exclude[start : start + chunk_size]] = -np.inf
            top = np.argpartition(-block, k - 1, axis=1)[:, :k]
            top_scores = np.take_along_axis(block, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            positions[start : start + len(block)] = np.take_along_axis(top, order, axis=1)
            scores[start : start + len(block)] = np.take_along_axis(top_scores, order, axis=1)
        return SearchResult(positions, scores)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, ids=self.ids, vectors=self.vectors.astype(np.float16), encoder=self.encoder
        )

    @classmethod
    def load(cls, path: Path) -> EmbeddingIndex:
        with np.load(path) as data:
            vectors = data["vectors"].astype(np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # undo fp16 rounding
            return cls(data["ids"], vectors, str(data["encoder"]))
