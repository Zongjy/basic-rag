"""
Document Chunking Module

Provides fixed-length document chunking with overlap support.
"""

import tiktoken
from typing import List, Dict, Any, Optional


class DocumentChunker:
    """
    Chunks documents into fixed-size pieces with optional overlap.

    Supports token-based and character-based chunking.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        mode: str = "token",  # "token" or "char"
        encoding_name: str = "cl100k_base"  # OpenAI's encoding
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of units to overlap between chunks
            mode: "token" for token-based, "char" for character-based
            encoding_name: Tokenizer encoding (for token mode)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode = mode

        if mode == "token":
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                print(f"Warning: Failed to load tokenizer ({e}), falling back to char mode")
                self.mode = "char"

    def chunk_text(self, text: str, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Chunk a single text into overlapping pieces.

        Args:
            text: Text to chunk
            doc_id: Optional document identifier

        Returns:
            List of chunk dictionaries with metadata
        """
        if self.mode == "token":
            return self._chunk_by_tokens(text, doc_id)
        else:
            return self._chunk_by_chars(text, doc_id)

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        id_field: str = "id"
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries
            text_field: Field name containing text
            id_field: Field name containing document ID

        Returns:
            List of all chunks with metadata
        """
        all_chunks = []

        for doc in documents:
            text = doc.get(text_field, "")
            doc_id = doc.get(id_field, None)

            if not text:
                continue

            chunks = self.chunk_text(text, doc_id)

            # Add original document metadata to each chunk
            for chunk in chunks:
                chunk["original_doc"] = {
                    k: v for k, v in doc.items()
                    if k not in [text_field]  # Exclude text to save memory
                }

            all_chunks.extend(chunks)

        return all_chunks

    def _chunk_by_tokens(self, text: str, doc_id: Optional[str]) -> List[Dict[str, Any]]:
        """Chunk text by token count."""
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        chunk_idx = 0

        while start < len(tokens):
            # Get chunk tokens
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}" if doc_id else f"chunk_{chunk_idx}",
                "text": chunk_text,
                "start_token": start,
                "end_token": end,
                "token_count": len(chunk_tokens),
                "doc_id": doc_id,
                "chunk_index": chunk_idx
            })

            # Move to next chunk with overlap
            start += self.chunk_size - self.chunk_overlap
            chunk_idx += 1

        return chunks

    def _chunk_by_chars(self, text: str, doc_id: Optional[str]) -> List[Dict[str, Any]]:
        """Chunk text by character count."""
        chunks = []

        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}" if doc_id else f"chunk_{chunk_idx}",
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "char_count": len(chunk_text),
                "doc_id": doc_id,
                "chunk_index": chunk_idx
            })

            start += self.chunk_size - self.chunk_overlap
            chunk_idx += 1

        return chunks

    def get_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about chunked documents.

        Args:
            chunks: List of chunks

        Returns:
            Statistics dictionary
        """
        total_chunks = len(chunks)
        unique_docs = len(set(c["doc_id"] for c in chunks if c["doc_id"]))

        if self.mode == "token":
            sizes = [c["token_count"] for c in chunks]
        else:
            sizes = [c["char_count"] for c in chunks]

        avg_size = sum(sizes) / len(sizes) if sizes else 0

        return {
            "total_chunks": total_chunks,
            "unique_documents": unique_docs,
            "avg_chunk_size": avg_size,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "chunking_mode": self.mode
        }
