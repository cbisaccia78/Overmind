"""Deterministic local vector memory.

Implements hashing-trick embeddings and cosine similarity search.
Persists items via the repository.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from .repository import Repository


class LocalVectorMemory:
    """Simple deterministic hashing-trick vector store."""

    def __init__(self, repo: Repository, dims: int = 128):
        """Initialize the memory store.

        Args:
            repo: Repository used to persist and retrieve memory items.
            dims: Embedding dimensionality for hashing-trick vectors.
        """
        self.repo = repo
        self.dims = dims

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize free-form text into lowercase alphanumeric tokens.

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokens.
        """
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def embed(self, text: str) -> list[float]:
        """Embed text into a deterministic unit-length vector.

        Uses a hashing trick over tokens and then L2-normalizes the vector.

        Args:
            text: Text to embed.

        Returns:
            A list of floats of length `dims`.
        """
        vec = [0.0] * self.dims
        for token in self._tokenize(text):
            idx = hash(token) % self.dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity for already-normalized vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Dot product between `a` and `b`.
        """
        return sum(x * y for x, y in zip(a, b))

    def store(
        self, text: str, collection: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Persist a memory item with an embedding.

        Args:
            text: The text content to store.
            collection: Logical collection name used for grouping.
            metadata: Optional metadata to store with the item.

        Returns:
            The stored item row (as a dict).
        """
        embedding = self.embed(text)
        return self.repo.add_memory_item(
            collection=collection, text=text, embedding=embedding, metadata=metadata
        )

    def search(
        self, query: str, collection: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search for the most similar memory items within a collection.

        Args:
            query: Query string to embed.
            collection: Collection to search.
            top_k: Maximum number of results to return.

        Returns:
            List of items enriched with a `score` field.
        """
        q = self.embed(query)
        items = self.repo.list_memory_items(collection)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            score = self._cosine(q, item["embedding"])
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, item in scored[:top_k]:
            enriched = defaultdict(lambda: None, item)
            enriched["score"] = score
            results.append(dict(enriched))
        return results
