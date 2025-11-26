"""
Vector Retrieval Module

Provides semantic search functionality using FAISS index.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class VectorRetriever:
    """
    Performs semantic search using FAISS index.

    Retrieves most similar chunks for given queries.
    """

    def __init__(
        self,
        index_builder,  # FAISSIndexBuilder instance
        embedding_client  # EmbeddingClient instance
    ):
        """
        Initialize retriever.

        Args:
            index_builder: FAISSIndexBuilder with built index
            embedding_client: EmbeddingClient for query embedding
        """
        self.index_builder = index_builder
        self.embedding_client = embedding_client

        if not index_builder.is_built:
            raise RuntimeError("Index not built. Build index before creating retriever.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = True,
        return_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks given a query.

        Args:
            query: Query text
            top_k: Number of results to return
            return_scores: Whether to include similarity scores
            return_embeddings: Whether to include chunk embeddings

        Returns:
            List of result dictionaries with metadata and optionally scores
        """
        # Embed query
        query_embedding = self.embedding_client.embed_texts([query], show_progress=False)

        # Normalize if using cosine similarity
        if self.index_builder.metric == "cosine":
            query_embedding = self._normalize_vectors(query_embedding)

        # Search
        scores, indices = self.index_builder.index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            result = {
                "rank": len(results),
                **self.index_builder.metadata[idx]  # Include all metadata
            }

            if return_scores:
                result["score"] = float(score)

            if return_embeddings:
                # Note: This would require storing embeddings separately
                result["embedding"] = None  # Placeholder

            results.append(result)

        return results

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Search multiple queries in batch.

        Args:
            queries: List of query texts
            top_k: Number of results per query
            return_scores: Whether to include similarity scores

        Returns:
            List of result lists, one per query
        """
        # Embed all queries
        query_embeddings = self.embedding_client.embed_texts(queries, show_progress=True)

        # Normalize if using cosine similarity
        if self.index_builder.metric == "cosine":
            query_embeddings = self._normalize_vectors(query_embeddings)

        # Search
        scores, indices = self.index_builder.index.search(query_embeddings, top_k)

        # Prepare results for each query
        all_results = []
        for query_idx, (query_indices, query_scores) in enumerate(zip(indices, scores)):
            results = []
            for idx, score in zip(query_indices, query_scores):
                if idx == -1:
                    continue

                result = {
                    "rank": len(results),
                    "query_index": query_idx,
                    **self.index_builder.metadata[idx]
                }

                if return_scores:
                    result["score"] = float(score)

                results.append(result)

            all_results.append(results)

        return all_results

    def search_with_filter(
        self,
        query: str,
        top_k: int = 5,
        filter_fn=None,  # Function to filter metadata
        max_candidates: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search with metadata filtering.

        Args:
            query: Query text
            top_k: Number of results to return (after filtering)
            filter_fn: Function that takes metadata dict and returns bool
            max_candidates: Number of candidates to retrieve before filtering

        Returns:
            Filtered results
        """
        if filter_fn is None:
            return self.search(query, top_k)

        # Retrieve more candidates for filtering
        candidates = self.search(query, top_k=max_candidates, return_scores=True)

        # Filter
        filtered = [c for c in candidates if filter_fn(c)]

        # Return top_k after filtering
        return filtered[:top_k]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve chunk metadata by chunk_id.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk metadata or None if not found
        """
        for meta in self.index_builder.metadata:
            if meta.get("chunk_id") == chunk_id:
                return meta
        return None

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "index_stats": self.index_builder.get_stats(),
            "embedding_info": self.embedding_client.get_info()
        }
