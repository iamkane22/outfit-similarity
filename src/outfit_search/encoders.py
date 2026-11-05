"""Image/text encoders producing L2-normalised embeddings in a shared space."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

logger = logging.getLogger(__name__)

Embeddings = NDArray[np.float32]
T = TypeVar("T")


class Encoder(Protocol):
    name: str

    def encode_images(self, images: Iterable[Image.Image]) -> Embeddings: ...

    def encode_texts(self, texts: Sequence[str]) -> Embeddings: ...


def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk


def normalise(vectors: NDArray[np.floating]) -> Embeddings:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit: Embeddings = (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)
    return unit


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenClipEncoder:
    """Any open_clip model, e.g. ``hf-hub:Marqo/marqo-fashionCLIP``."""

    def __init__(
        self, name: str, model_id: str, device: str | None = None, batch_size: int = 64
    ) -> None:
        import open_clip
        import torch

        self.name = name
        self.device = device or pick_device()
        self.batch_size = batch_size
        model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_id)
        self._torch = torch
        logger.info("Loaded %s on %s", model_id, self.device)

    def encode_images(self, images: Iterable[Image.Image]) -> Embeddings:
        torch = self._torch
        chunks = []
        done = 0
        with torch.inference_mode():
            for step, batch in enumerate(batched(images, self.batch_size), start=1):
                pixels = torch.stack([self._preprocess(img) for img in batch]).to(self.device)
                chunks.append(self._model.encode_image(pixels).float().cpu().numpy())
                done += len(batch)
                if step % 20 == 0:
                    logger.info("%s: embedded %d images", self.name, done)
        return normalise(np.concatenate(chunks)) if chunks else np.empty((0, 0), np.float32)

    def encode_texts(self, texts: Sequence[str]) -> Embeddings:
        torch = self._torch
        chunks = []
        with torch.inference_mode():
            for batch in batched(texts, 256):
                tokens = self._tokenizer(batch).to(self.device)
                chunks.append(self._model.encode_text(tokens).float().cpu().numpy())
        return normalise(np.concatenate(chunks)) if chunks else np.empty((0, 0), np.float32)
