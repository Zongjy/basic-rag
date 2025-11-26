"""
Basic RAG System - Main Module

Provides high-level interface integrating all components.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .chunking import DocumentChunker
from .embedding import EmbeddingClient
from .indexing import FAISSIndexBuilder
from .retrieval import VectorRetriever


class RAGSystem:
    """
    Complete RAG system integrating chunking, embedding, indexing, and retrieval.

    Simple API for document indexing and semantic search.
    """

    def __init__(
        self,
        # Embedding config
        embedding_api_url: str,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: Optional[str] = None,
        embedding_batch_size: int = 32,
        # Chunking config
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        chunk_mode: str = "token",  # "token" or "char"
        # Index config
        index_type: str = "flat",  # "flat" or "hnsw"
        metric: str = "cosine",  # "cosine", "l2", "ip"
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        # Other
        verbose: bool = True
    ):
        """
        Initialize RAG system.

        Args:
            embedding_api_url: URL for embedding API
            embedding_model: Model name for embeddings
            embedding_api_key: Optional API key
            embedding_batch_size: Batch size for embedding requests
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chunk_mode: "token" or "char" based chunking
            index_type: FAISS index type ("flat" or "hnsw")
            metric: Distance metric
            hnsw_m: HNSW parameter
            hnsw_ef_construction: HNSW construction parameter
            hnsw_ef_search: HNSW search parameter
            verbose: Print progress messages
        """
        self.verbose = verbose

        # Initialize chunker
        self._log("🔧 Initializing document chunker...")
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            mode=chunk_mode
        )

        # Initialize embedding client
        self._log("🔧 Initializing embedding client...")
        self.embedding_client = EmbeddingClient(
            api_url=embedding_api_url,
            model_name=embedding_model,
            api_key=embedding_api_key,
            batch_size=embedding_batch_size
        )

        # Index builder (will be initialized after getting embedding dim)
        self.index_builder = None
        self._index_config = {
            "index_type": index_type,
            "metric": metric,
            "hnsw_m": hnsw_m,
            "hnsw_ef_construction": hnsw_ef_construction,
            "hnsw_ef_search": hnsw_ef_search
        }

        # Retriever (will be initialized after building index)
        self.retriever = None

        # Store indexed chunks for reference
        self.indexed_chunks = []

        self._log("✅ RAG system initialized")

    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(message)

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        id_field: str = "id"
    ) -> Dict[str, Any]:
        """
        Index documents: chunk, embed, and build FAISS index.

        Args:
            documents: List of document dicts
            text_field: Field name containing text
            id_field: Field name containing document ID

        Returns:
            Statistics about indexing process
        """
        self._log("=" * 60)
        self._log("📚 Starting document indexing...")
        self._log("=" * 60)

        # Step 1: Chunk documents
        self._log("\n📄 Step 1: Chunking documents...")
        chunks = self.chunker.chunk_documents(documents, text_field, id_field)
        chunk_stats = self.chunker.get_stats(chunks)
        self._log(f"  Created {chunk_stats['total_chunks']} chunks from {chunk_stats['unique_documents']} documents")

        # Step 2: Embed chunks
        self._log("\n🔢 Step 2: Embedding chunks...")
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_client.embed_texts(chunk_texts, show_progress=self.verbose)
        self._log(f"  Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

        # Step 3: Build FAISS index
        self._log("\n🏗️  Step 3: Building FAISS index...")

        # Initialize index builder if needed
        if self.index_builder is None:
            embedding_dim = embeddings.shape[1]
            self.index_builder = FAISSIndexBuilder(
                embedding_dim=embedding_dim,
                **self._index_config
            )

        # Build index
        self.index_builder.build_index(embeddings, chunks)

        # Step 4: Initialize retriever
        self._log("\n🔍 Step 4: Initializing retriever...")
        self.retriever = VectorRetriever(self.index_builder, self.embedding_client)

        # Store chunks
        self.indexed_chunks = chunks

        self._log("\n" + "=" * 60)
        self._log("✅ Indexing complete!")
        self._log("=" * 60)

        return {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "chunk_stats": chunk_stats,
            "embedding_dim": embeddings.shape[1],
            "index_type": self._index_config["index_type"]
        }

    def index_from_file(
        self,
        file_path: str,
        text_field: str = "text",
        id_field: str = "id"
    ) -> Dict[str, Any]:
        """
        Index documents from JSONL file.

        Args:
            file_path: Path to JSONL file
            text_field: Field containing text
            id_field: Field containing ID

        Returns:
            Indexing statistics
        """
        self._log(f"📖 Loading documents from {file_path}...")

        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        self._log(f"  Loaded {len(documents)} documents")

        return self.index_documents(documents, text_field, id_field)

    def search(
        self,
        query: Union[str, List[str]],
        top_k: int = 5,
        return_scores: bool = True
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Search for relevant chunks.

        Args:
            query: Single query or list of queries
            top_k: Number of results per query
            return_scores: Include similarity scores

        Returns:
            Results (single list if query is str, list of lists if query is list)
        """
        if self.retriever is None:
            raise RuntimeError("Index not built. Call index_documents() first.")

        if isinstance(query, str):
            return self.retriever.search(query, top_k, return_scores)
        else:
            return self.retriever.batch_search(query, top_k, return_scores)

    def save(self, save_dir: str) -> None:
        """
        Save index and metadata to directory.

        Args:
            save_dir: Directory to save files
        """
        if self.index_builder is None:
            raise RuntimeError("No index to save. Call index_documents() first.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index and metadata
        index_path = str(save_path / "index.faiss")
        self.index_builder.save(index_path)

        # Save chunks for reference
        chunks_path = save_path / "chunks.jsonl"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            for chunk in self.indexed_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        self._log(f"✅ RAG system saved to {save_dir}")

    def load(self, load_dir: str) -> None:
        """
        Load index and metadata from directory.

        Args:
            load_dir: Directory containing saved files
        """
        load_path = Path(load_dir)

        # Load FAISS index
        index_path = str(load_path / "index.faiss")

        # Initialize index builder if needed
        if self.index_builder is None:
            # Create placeholder, will be configured from loaded metadata
            self.index_builder = FAISSIndexBuilder(embedding_dim=768, **self._index_config)

        self.index_builder.load(index_path)

        # Load chunks
        chunks_path = load_path / "chunks.jsonl"
        self.indexed_chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                self.indexed_chunks.append(chunk)

        # Initialize retriever
        self.retriever = VectorRetriever(self.index_builder, self.embedding_client)

        self._log(f"✅ RAG system loaded from {load_dir}")

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "chunker_config": {
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap,
                "mode": self.chunker.mode
            },
            "embedding_info": self.embedding_client.get_info()
        }

        if self.index_builder:
            stats["index_stats"] = self.index_builder.get_stats()

        if self.indexed_chunks:
            stats["indexed_chunks"] = len(self.indexed_chunks)

        return stats
