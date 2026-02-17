"""Vector memory with pluggable embedding backends and indexed retrieval."""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from urllib import error as urlerror
from urllib import request as urlrequest
import json
from typing import Any

from .repository import Repository


class LocalVectorMemory:
    """Vector memory that supports API/local embedding providers."""

    def __init__(self, repo: Repository):
        """Initialize the memory store.

        Args:
            repo: Repository used to persist and retrieve memory items.
        """
        self.repo = repo
        self.backend = _select_backend()

    def embed(self, text: str) -> list[float]:
        """Embed text via the configured embedding backend."""
        return self.backend.embed(text)

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
            collection=collection,
            text=text,
            embedding=embedding,
            embedding_model=self.backend.model_name,
            dims=self.backend.dims,
            metadata=metadata,
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
        candidate_limit = max(top_k * 20, 50)
        items = self.repo.search_memory_candidates(
            collection=collection,
            query=query,
            limit=candidate_limit,
        )
        if len(items) < top_k:
            existing_ids = {item["id"] for item in items}
            for item in self.repo.list_memory_items(collection):
                if item["id"] not in existing_ids:
                    items.append(item)
                    existing_ids.add(item["id"])
                if len(items) >= top_k:
                    break
        if not items:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            if int(item.get("dims") or 0) != len(q):
                continue
            score = self._cosine(q, item["embedding"])
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, item in scored[:top_k]:
            enriched = defaultdict(lambda: None, item)
            enriched["score"] = score
            results.append(dict(enriched))
        return results


class EmbeddingBackend(ABC):
    """Interface for embedding providers."""

    model_name: str
    dims: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a normalized embedding vector."""


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI embeddings API backend."""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.dims = int(os.getenv("OVERMIND_EMBEDDING_DIMS", "1536"))
        self.endpoint = os.getenv(
            "OVERMIND_OPENAI_EMBEDDINGS_URL",
            "https://api.openai.com/v1/embeddings",
        )

    def embed(self, text: str) -> list[float]:
        payload = {
            "model": self.model_name,
            "input": text,
        }
        req = urlrequest.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urlerror.URLError, TimeoutError) as exc:
            raise RuntimeError(f"embedding request failed: {exc}") from exc

        embedding = data["data"][0]["embedding"]
        self.dims = len(embedding)
        return _normalize([float(x) for x in embedding])


class LocalBm25Backend(EmbeddingBackend):
    """Local lexical backend that pairs with FTS/BM25 retrieval."""

    model_name = "fts5-bm25"
    dims = 0

    def embed(self, text: str) -> list[float]:
        del text
        return []


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _select_backend() -> EmbeddingBackend:
    provider = os.getenv("OVERMIND_EMBEDDING_PROVIDER", "auto").lower()
    if provider in {"auto", "openai"}:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            model_name = os.getenv("OVERMIND_EMBEDDING_MODEL", "text-embedding-3-small")
            return OpenAIEmbeddingBackend(api_key=api_key, model_name=model_name)
        if provider == "openai":
            raise RuntimeError(
                "OPENAI_API_KEY is required for OVERMIND_EMBEDDING_PROVIDER=openai"
            )
    return LocalBm25Backend()
