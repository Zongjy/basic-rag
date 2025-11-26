"""
Basic RAG System

A simple, modular RAG system with chunking, embedding, indexing, and retrieval.
"""

from .rag_system import RAGSystem
from .chunking import DocumentChunker
from .embedding import EmbeddingClient
from .indexing import FAISSIndexBuilder
from .retrieval import VectorRetriever

__version__ = "0.1.0"

__all__ = [
    "RAGSystem",
    "DocumentChunker",
    "EmbeddingClient",
    "FAISSIndexBuilder",
    "VectorRetriever"
]
