from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from outfit_search.encoders import normalise
from outfit_search.evaluation import evaluate, item_to_item, text_to_image, zero_shot
from outfit_search.index import EmbeddingIndex
from tests.conftest import FakeEncoder


def test_item_to_item_is_perfect_when_labels_form_tight_clusters() -> None:
    labels = np.repeat(np.arange(4), 25)
    centres = normalise(np.eye(4, 8))
    vectors = normalise(centres[labels] + np.random.default_rng(0).normal(0, 0.01, (100, 8)))
    items = pd.DataFrame({"articleType": labels.astype(str), "baseColour": "Black"})
    index = EmbeddingIndex(np.arange(100), vectors, "toy")
    metrics = item_to_item(index, items, np.arange(100), k=10)
    assert metrics == pytest.approx({"precision@10": 1.0, "hit@1": 1.0, "map@10": 1.0})


def test_item_to_item_is_near_chance_for_random_vectors() -> None:
    labels = np.repeat(np.arange(10), 30)
    vectors = normalise(np.random.default_rng(1).normal(size=(300, 32)))
    items = pd.DataFrame({"articleType": labels.astype(str), "baseColour": "Black"})
    index = EmbeddingIndex(np.arange(300), vectors, "noise")
    assert item_to_item(index, items, np.arange(300))["precision@10"] < 0.2


def test_zero_shot_colour_with_fake_encoder(index: EmbeddingIndex, items: pd.DataFrame) -> None:
    metrics = zero_shot(index, FakeEncoder(), items, "baseColour", "a {} item", min_count=1)
    assert metrics["accuracy"] == 1.0
    assert metrics["classes"] == 3


def test_text_to_image_ranks(index: EmbeddingIndex) -> None:
    metrics = text_to_image(index, FakeEncoder(), ["red"], np.array([index.position(102)]))
    assert metrics["recall@10"] == 1.0
    assert 0.0 < metrics["mrr"] <= 1.0


def test_evaluate_requires_aligned_inputs(index: EmbeddingIndex, items: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="aligned"):
        evaluate(index, FakeEncoder(), items.iloc[::-1])


def test_chance_precision_matches_the_definition() -> None:
    from outfit_search.evaluation import chance_precision

    items = pd.DataFrame({"articleType": ["A"] * 3 + ["B"] * 2, "baseColour": "Red"})
    # pairs sharing a label: 3*2 + 2*1 = 8 of 5*4 = 20 ordered pairs
    assert chance_precision(items) == pytest.approx(8 / 20)


def test_paired_bootstrap_interval_contains_the_mean() -> None:
    from outfit_search.evaluation import paired_bootstrap

    rng = np.random.default_rng(0)
    a = rng.normal(0.35, 0.1, 500)
    b = a - 0.03 + rng.normal(0, 0.01, 500)
    mean, low, high = paired_bootstrap(a, b)
    assert mean == pytest.approx(0.03, abs=0.002)
    assert low < mean < high
    assert low > 0
