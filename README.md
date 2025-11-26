# Basic RAG System

A simple, modular RAG (Retrieval-Augmented Generation) system with:

- **Chunking**: Fixed-length document splitting with overlap
- **Embedding**: Remote API integration for vector embeddings
- **Indexing**: FAISS-based vector index construction
- **Retrieval**: Efficient semantic search

## Architecture

```
basic_rag/
├── chunking/       # Document chunking module
├── embedding/      # Embedding API client
├── indexing/       # FAISS index construction
├── retrieval/      # Vector retrieval
└── utils/          # Shared utilities
```

## Quick Start

```python
from basic_rag import RAGSystem

# Initialize RAG system
rag = RAGSystem(
    embedding_api_url="http://localhost:30000/v1/embeddings",
    embedding_model="text-embedding-3-small",
    chunk_size=512,
    chunk_overlap=50
)

# Index documents
rag.index_documents([
    {"id": "doc1", "text": "Your document text here..."},
    {"id": "doc2", "text": "Another document..."}
])

# Search
results = rag.search("your query", top_k=5)
```

## Installation

```bash
cd basic-rag
uv venv
source .venv/bin/activate
uv pip install -e .
```
