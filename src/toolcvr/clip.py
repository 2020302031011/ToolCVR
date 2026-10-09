from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from PIL import Image


class ClipEncoder(Protocol):
    @property
    def identity(self) -> str:
        """Stable model/preprocessor identity for cache invalidation."""

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return an L2-normalized [N,D] embedding matrix."""

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return an L2-normalized [N,D] embedding matrix."""


def _normalize(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("embedding output must be a matrix")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norms, 1e-12)


class TransformersClipEncoder:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cuda:0",
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("CLIP batch size must be positive")
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError("real CLIP requires torch and transformers") from exc

        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = batch_size
        dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_name_or_path)
        self._model = CLIPModel.from_pretrained(model_name_or_path, dtype=dtype)
        self._model.to(self._device).eval()
        self._identity = f"transformers-clip:{model_name_or_path}"

    @property
    def identity(self) -> str:
        return self._identity

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, self._model.config.projection_dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, len(images), self._batch_size):
            batch = [Image.fromarray(item, mode="RGB") for item in images[start : start + self._batch_size]]
            inputs = self._processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                values = self._model.get_image_features(**inputs)
            chunks.append(values.detach().float().cpu().numpy())
        return _normalize(np.concatenate(chunks, axis=0))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._model.config.projection_dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            inputs = self._processor(text=batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                values = self._model.get_text_features(**inputs)
            chunks.append(values.detach().float().cpu().numpy())
        return _normalize(np.concatenate(chunks, axis=0))


class DeterministicMockClipEncoder:
    """Cheap deterministic encoder for CPU integration tests.

    It is intentionally not semantically meaningful. The selector will often
    enter its flat-query fallback, which is useful for exercising that path.
    """

    dimension = 12

    @property
    def identity(self) -> str:
        return "mock-clip-v1"

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for image in images:
            value = image.astype(np.float32) / 255.0
            means = value.mean(axis=(0, 1))
            stds = value.std(axis=(0, 1))
            q25 = np.quantile(value, 0.25, axis=(0, 1))
            q75 = np.quantile(value, 0.75, axis=(0, 1))
            rows.append(np.concatenate([means, stds, q25, q75]).astype(np.float32))
        if not rows:
            return np.empty((0, self.dimension), dtype=np.float32)
        return _normalize(np.stack(rows))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            row = np.frombuffer(digest[: self.dimension], dtype=np.uint8).astype(np.float32) + 1.0
            rows.append(row)
        if not rows:
            return np.empty((0, self.dimension), dtype=np.float32)
        return _normalize(np.stack(rows))
