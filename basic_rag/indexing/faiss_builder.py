"""
FAISS Index Builder Module

Provides FAISS index construction and management for vector search.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class FAISSIndexBuilder:
    """
    Builds and manages FAISS indices for efficient vector search.

    Supports flat (exact) and HNSW (approximate) indices.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",  # "flat" or "hnsw"
        metric: str = "cosine",  # "cosine", "l2", or "ip" (inner product)
        hnsw_m: int = 32,  # HNSW parameter
        hnsw_ef_construction: int = 200,  # HNSW construction parameter
        hnsw_ef_search: int = 64  # HNSW search parameter
    ):
        """
        Initialize index builder.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ("flat" for exact, "hnsw" for approximate)
            metric: Distance metric
            hnsw_m: Number of connections per layer (HNSW only)
            hnsw_ef_construction: Size of dynamic candidate list (HNSW only)
            hnsw_ef_search: Search depth (HNSW only)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        self.index = None
        self.metadata = []  # Store chunk metadata
        self.is_built = False

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            metadata: List of metadata dicts for each embedding
        """
        if len(embeddings) != len(metadata):
            raise ValueError(f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) length mismatch")

        print(f"Building FAISS index ({self.index_type})...")
        print(f"  Vectors: {len(embeddings)}")
        print(f"  Dimension: {self.embedding_dim}")
        print(f"  Metric: {self.metric}")

        # Normalize for cosine similarity
        if self.metric == "cosine":
            embeddings = self._normalize_vectors(embeddings)

        # Create index based on type
        if self.index_type == "flat":
            self.index = self._create_flat_index(embeddings)
        elif self.index_type == "hnsw":
            self.index = self._create_hnsw_index(embeddings)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # Store metadata
        self.metadata = metadata
        self.is_built = True

        print(f"✅ Index built with {self.index.ntotal} vectors")

    def _create_flat_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create flat (exact) index."""
        if self.metric == "cosine":
            # After normalization, cosine similarity = inner product
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.metric == "l2":
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.metric == "ip":
            index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        index.add(embeddings)
        return index

    def _create_hnsw_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create HNSW (approximate) index."""
        if self.metric == "cosine":
            index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        elif self.metric == "l2":
            index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m, faiss.METRIC_L2)
        else:
            raise ValueError(f"Unsupported metric for HNSW: {self.metric}")

        # Set construction parameters
        index.hnsw.efConstruction = self.hnsw_ef_construction

        # Add vectors
        index.add(embeddings)

        # Set search parameter
        index.hnsw.efSearch = self.hnsw_ef_search

        return index

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Add more vectors to existing index.

        Args:
            embeddings: New embeddings to add
            metadata: Metadata for new embeddings
        """
        if not self.is_built:
            raise RuntimeError("Index not built yet. Call build_index first.")

        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata length mismatch")

        # Normalize if using cosine
        if self.metric == "cosine":
            embeddings = self._normalize_vectors(embeddings)

        # Add to index
        self.index.add(embeddings)
        self.metadata.extend(metadata)

        print(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")

    def save(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save index and metadata to disk.

        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata (defaults to index_path + '.meta')
        """
        if not self.is_built:
            raise RuntimeError("Index not built yet. Call build_index first.")

        # Save FAISS index
        faiss.write_index(self.index, index_path)
        print(f"✅ Index saved to {index_path}")

        # Save metadata
        if metadata_path is None:
            metadata_path = index_path + ".meta"

        with open(metadata_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "config": {
                    "embedding_dim": self.embedding_dim,
                    "index_type": self.index_type,
                    "metric": self.metric,
                    "hnsw_m": self.hnsw_m,
                    "hnsw_ef_construction": self.hnsw_ef_construction,
                    "hnsw_ef_search": self.hnsw_ef_search
                }
            }, f)
        print(f"✅ Metadata saved to {metadata_path}")

    def load(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load index and metadata from disk.

        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        print(f"✅ Index loaded from {index_path}")

        # Load metadata
        if metadata_path is None:
            metadata_path = index_path + ".meta"

        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            config = data["config"]

            # Restore configuration
            self.embedding_dim = config["embedding_dim"]
            self.index_type = config["index_type"]
            self.metric = config["metric"]
            self.hnsw_m = config.get("hnsw_m", 32)
            self.hnsw_ef_construction = config.get("hnsw_ef_construction", 200)
            self.hnsw_ef_search = config.get("hnsw_ef_search", 64)

        self.is_built = True
        print(f"✅ Metadata loaded from {metadata_path}")
        print(f"  Total vectors: {self.index.ntotal}")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.is_built:
            return {"status": "not_built"}

        return {
            "status": "built",
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metric": self.metric,
            "metadata_count": len(self.metadata)
        }
