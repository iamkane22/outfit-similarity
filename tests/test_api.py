from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from outfit_search.api import create_app
from outfit_search.index import EmbeddingIndex
from tests.conftest import COLOURS, FakeEncoder


@pytest.fixture
def client(index: EmbeddingIndex, items: pd.DataFrame) -> TestClient:
    return TestClient(create_app(index, items, FakeEncoder()))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "encoder": "fake", "items": 12}


def test_text_search_finds_the_colour(client: TestClient) -> None:
    hits = client.post("/search/text", json={"query": "red jeans", "k": 4}).json()["hits"]
    assert len(hits) == 4
    assert all(hit["colour"] == "Red" for hit in hits)


def test_filters_are_applied_inside_the_index(client: TestClient) -> None:
    body = {"query": "navy", "k": 50, "gender": "Women"}
    hits = client.post("/search/text", json=body).json()["hits"]
    assert len(hits) == 6
    names = [hit["name"] for hit in hits]
    assert all(name.startswith("Women") for name in names)


def test_more_like_this_excludes_the_item_itself(client: TestClient) -> None:
    hits = client.get("/items/100/similar", params={"k": 3}).json()["hits"]
    assert 100 not in [hit["id"] for hit in hits]
    assert hits[0]["colour"] == "Navy Blue"


def test_unknown_item_is_404(client: TestClient) -> None:
    assert client.get("/items/999/similar").status_code == 404


def test_image_search(client: TestClient) -> None:
    buffer = io.BytesIO()
    Image.fromarray(np.full((10, 10, 3), COLOURS["White"], dtype=np.uint8)).save(buffer, "PNG")
    response = client.post(
        "/search/image", files={"file": ("q.png", buffer.getvalue(), "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["hits"][0]["colour"] == "White"


def test_rejects_non_images(client: TestClient) -> None:
    response = client.post("/search/image", files={"file": ("q.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_validation_errors(client: TestClient) -> None:
    assert client.post("/search/text", json={"query": ""}).status_code == 422
    assert client.post("/search/text", json={"query": "x", "k": 1000}).status_code == 422
