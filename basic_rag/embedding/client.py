"""
Embedding API Client Module

Provides async/sync HTTP clients for getting text embeddings from remote APIs.
"""

import asyncio
import aiohttp
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm


class EmbeddingClient:
    """
    Client for getting embeddings from remote API.

    Supports OpenAI-compatible API format.
    """

    def __init__(
        self,
        api_url: str,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 32,
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize embedding client.

        Args:
            api_url: API endpoint URL (e.g., "http://localhost:30000/v1/embeddings")
            model_name: Model name to use
            api_key: Optional API key for authentication
            batch_size: Number of texts to embed in one request
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

        # Store embedding dimension (will be set after first call)
        self.embedding_dim = None

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed multiple texts synchronously.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []

        # Process in batches
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Embedding texts",
            disable=not show_progress
        ):
            batch = texts[i:i + self.batch_size]
            embeddings = self._embed_batch_sync(batch)
            all_embeddings.extend(embeddings)

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Set embedding dimension
        if self.embedding_dim is None and len(embeddings_array) > 0:
            self.embedding_dim = embeddings_array.shape[1]

        return embeddings_array

    async def embed_texts_async(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed multiple texts asynchronously.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        # Create batches
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        # Process batches concurrently
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_batch_async(session, batch) for batch in batches]

            if show_progress:
                # With progress bar
                all_embeddings = []
                for task in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc="Embedding texts"
                ):
                    embeddings = await task
                    all_embeddings.extend(embeddings)
            else:
                # Without progress bar
                results = await asyncio.gather(*tasks)
                all_embeddings = [emb for batch_embs in results for emb in batch_embs]

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Set embedding dimension
        if self.embedding_dim is None and len(embeddings_array) > 0:
            self.embedding_dim = embeddings_array.shape[1]

        return embeddings_array

    def _embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts synchronously."""
        payload = {
            "input": texts,
            "model": self.model_name
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to get embeddings after {self.max_retries} attempts: {e}")
                print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")
                continue

    async def _embed_batch_async(
        self,
        session: aiohttp.ClientSession,
        texts: List[str]
    ) -> List[List[float]]:
        """Embed a batch of texts asynchronously."""
        payload = {
            "input": texts,
            "model": self.model_name
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    return embeddings

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to get embeddings after {self.max_retries} attempts: {e}")
                print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")
                await asyncio.sleep(1)
                continue

    def get_info(self) -> Dict[str, Any]:
        """Get client information."""
        return {
            "api_url": self.api_url,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "has_api_key": self.api_key is not None
        }
