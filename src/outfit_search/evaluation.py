"""Measured retrieval quality.

The catalogue has no human relevance judgements, so we use its own labels:

* **Item-to-item search** - a result is relevant if it has the same ``articleType`` and
  ``baseColour`` as the query ("more navy shirts like this one"). Reported as
  precision@10, hit-rate@1 and mean average precision@10.
* **Zero-shot classification** - each item is assigned the label whose text prompt is
  closest ("a photo of a kurta"). Measures how well images and words share one space.
* **Text-to-image search** - the product title is used as a query; we record the rank of
  that exact product (recall@1, recall@10, mean reciprocal rank).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from outfit_search.encoders import Encoder
from outfit_search.index import EmbeddingIndex


def _labels(items: pd.DataFrame) -> NDArray[np.object_]:
    labels: NDArray[np.object_] = (
        items["articleType"] + "|" + items["baseColour"].fillna("")
    ).to_numpy()
    return labels


def relevance(
    index: EmbeddingIndex, items: pd.DataFrame, query_positions: NDArray[np.intp], k: int = 10
) -> NDArray[np.bool_]:
    """``(n_queries, k)`` matrix: is the result at each rank the same type and colour?"""
    labels = _labels(items)
    result = index.search(index.vectors[query_positions], k, exclude=query_positions)
    relevant: NDArray[np.bool_] = labels[result.positions] == labels[query_positions][:, None]
    return relevant


def chance_precision(items: pd.DataFrame) -> float:
    """Expected precision of a random ranking: P(random other item shares type and colour)."""
    counts = pd.Series(_labels(items)).value_counts().to_numpy(dtype=float)
    n = counts.sum()
    return float(np.sum(counts * (counts - 1)) / (n * (n - 1)))


def paired_bootstrap(
    a: NDArray[np.floating], b: NDArray[np.floating], n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``a - b`` over queries with a 95% percentile bootstrap interval."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_resamples, len(diff)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def item_to_item(
    index: EmbeddingIndex, items: pd.DataFrame, query_positions: NDArray[np.intp], k: int = 10
) -> dict[str, float]:
    labels = _labels(items)
    relevant = relevance(index, items, query_positions, k)

    hits_so_far = np.cumsum(relevant, axis=1)
    precision_at_rank = hits_so_far / np.arange(1, k + 1)
    n_relevant = np.minimum(
        pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy()[query_positions] - 1, k
    )
    ap = (precision_at_rank * relevant).sum(axis=1) / np.clip(n_relevant, 1, None)
    return {
        f"precision@{k}": float(relevant.mean()),
        "hit@1": float(relevant[:, 0].mean()),
        f"map@{k}": float(ap.mean()),
    }


def zero_shot(
    index: EmbeddingIndex,
    encoder: Encoder,
    items: pd.DataFrame,
    field: str,
    template: str,
    *,
    min_count: int = 50,
) -> dict[str, float]:
    counts = items[field].value_counts()
    classes = sorted(counts[counts >= min_count].index)
    mask = items[field].isin(classes).to_numpy()
    prompts = encoder.encode_texts([template.format(c.lower()) for c in classes])
    predicted = np.asarray(classes)[np.argmax(index.vectors[mask] @ prompts.T, axis=1)]
    truth = items.loc[mask, field].to_numpy()

    f1 = []
    for c in classes:
        tp = np.sum((predicted == c) & (truth == c))
        precision = tp / max(np.sum(predicted == c), 1)
        recall = tp / max(np.sum(truth == c), 1)
        f1.append(0.0 if tp == 0 else 2 * precision * recall / (precision + recall))
    return {
        "accuracy": float(np.mean(predicted == truth)),
        "macro_f1": float(np.mean(f1)),
        "classes": float(len(classes)),
    }


def text_to_image(
    index: EmbeddingIndex, encoder: Encoder, titles: Sequence[str], positions: NDArray[np.intp]
) -> dict[str, float]:
    queries = encoder.encode_texts(list(titles))
    rank = np.empty(len(positions), dtype=np.int64)
    for start in range(0, len(positions), 512):
        scores = queries[start : start + 512] @ index.vectors.T
        rows = np.arange(len(scores))
        target = scores[rows, positions[start : start + 512]]
        rank[start : start + len(scores)] = 1 + np.sum(scores > target[:, None], axis=1)
    return {
        "recall@1": float(np.mean(rank <= 1)),
        "recall@10": float(np.mean(rank <= 10)),
        "mrr": float(np.mean(1.0 / rank)),
    }


def evaluate(
    index: EmbeddingIndex,
    encoder: Encoder,
    items: pd.DataFrame,
    n_queries: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """All metrics for one encoder; ``items`` must be aligned with ``index``."""
    if not np.array_equal(items["id"].to_numpy(), index.ids):
        raise ValueError("items and index are not aligned")
    rng = np.random.default_rng(seed)
    queries = rng.choice(len(index), size=min(n_queries, len(index)), replace=False)
    metrics: dict[str, float] = {}
    metrics |= {f"item2item_{k}": v for k, v in item_to_item(index, items, queries).items()}
    metrics |= {
        f"zeroshot_type_{k}": v
        for k, v in zero_shot(index, encoder, items, "articleType", "a photo of a {}").items()
    }
    metrics |= {
        f"zeroshot_colour_{k}": v
        for k, v in zero_shot(
            index, encoder, items, "baseColour", "a photo of a {} fashion item"
        ).items()
    }
    titles = items["productDisplayName"].fillna("").to_numpy()[queries].tolist()
    metrics |= {
        f"text2image_{k}": v for k, v in text_to_image(index, encoder, titles, queries).items()
    }
    return metrics
