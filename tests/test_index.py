from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from outfit_search.encoders import normalise
from outfit_search.index import EmbeddingIndex


@pytest.fixture
def random_index() -> EmbeddingIndex:
    vectors = normalise(np.random.default_rng(0).normal(size=(500, 16)))
    return EmbeddingIndex(np.arange(1000, 1500), vectors, "random")


def test_matches_brute_force(random_index: EmbeddingIndex) -> None:
    queries = normalise(np.random.default_rng(1).normal(size=(37, 16)))
    result = random_index.search(queries, k=7, chunk_size=10)
    expected = np.argsort(-(queries @ random_index.vectors.T), axis=1)[:, :7]
    np.testing.assert_array_equal(result.positions, expected)
    assert np.all(np.diff(result.scores, axis=1) <= 0)


def test_exclude_drops_the_query_item(random_index: EmbeddingIndex) -> None:
    positions = np.arange(20)
    result = random_index.search(random_index.vectors[positions], k=5, exclude=positions)
    assert not np.any(result.positions == positions[:, None])


def test_candidate_mask_restricts_results(random_index: EmbeddingIndex) -> None:
    mask = np.zeros(len(random_index), dtype=bool)
    mask[::3] = True
    result = random_index.search(random_index.vectors[:4], k=10, candidates=mask)
    assert np.all(mask[result.positions])


def test_k_is_clamped_to_index_size() -> None:
    index = EmbeddingIndex(np.array([1, 2]), normalise(np.eye(2)), "tiny")
    assert index.search(np.eye(2, dtype=np.float32), k=10).positions.shape == (2, 2)


def test_save_and_load_round_trip(tmp_path: Path, random_index: EmbeddingIndex) -> None:
    path = tmp_path / "idx.npz"
    random_index.save(path)
    loaded = EmbeddingIndex.load(path)
    assert loaded.encoder == "random"
    np.testing.assert_array_equal(loaded.ids, random_index.ids)
    np.testing.assert_allclose(loaded.vectors, random_index.vectors, atol=2e-3)
    assert loaded.position(1042) == 42


def test_rejects_unnormalised_vectors() -> None:
    with pytest.raises(ValueError, match="normalised"):
        EmbeddingIndex(np.array([1]), np.array([[3.0, 4.0]], dtype=np.float32), "bad")
