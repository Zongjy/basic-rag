"""
MultiHop-RAG Dataset Evaluation Script

This script:
1. Loads the MultiHop-RAG dataset from HuggingFace
2. Chunks the corpus body field, with title+metadata as a separate chunk
3. Builds a FAISS index for semantic search
4. Searches for relevant chunks using queries
5. Saves output files in TaoTie format:
   - chunks.jsonl: All document chunks
   - index files: FAISS index and chunk_id mapping
   - multihop_rag.jsonl: Queries in TaoTie format (id, question, answer, contexts)
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm

# Import basic_rag modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from basic_rag.chunking.chunker import DocumentChunker
from basic_rag.embedding.client import EmbeddingClient
from basic_rag.indexing.faiss_builder import FAISSIndexBuilder
from basic_rag.retrieval.retriever import VectorRetriever


class MultiHopRAGEvaluator:
    """MultiHop-RAG dataset evaluator for RAG system."""

    def __init__(
        self,
        output_dir: str = "./multihop_rag_output",
        embedding_api_url: str = "http://localhost:30000/v1/embeddings",
        embedding_model: str = "gte-Qwen2-7B-Instruct",
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        top_k: int = 5
    ):
        """
        Initialize evaluator.

        Args:
            output_dir: Directory to save output files
            embedding_api_url: URL for embedding API
            embedding_model: Model name for embeddings
            chunk_size: Size of text chunks (in tokens)
            chunk_overlap: Overlap between chunks (in tokens)
            top_k: Number of chunks to retrieve per query
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.top_k = top_k

        # Initialize chunker
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            mode="token"
        )

        # Initialize embedding client
        self.embedding_client = EmbeddingClient(
            api_url=embedding_api_url,
            model_name=embedding_model,
            batch_size=32
        )

        self.index_builder = None
        self.retriever = None

    def load_multihop_dataset(self) -> tuple:
        """
        Load MultiHop-RAG dataset from HuggingFace.

        Returns:
            Tuple of (corpus_dataset, queries_dataset)
        """
        print("Loading MultiHop-RAG corpus dataset...")
        corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")
        corpus_data = corpus["train"]
        print(f"Loaded {len(corpus_data)} corpus documents")

        print("Loading MultiHop-RAG queries dataset...")
        queries = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
        queries_data = queries["train"]
        print(f"Loaded {len(queries_data)} queries")

        return corpus_data, queries_data

    def create_chunks_from_document(self, doc: Dict[str, Any], doc_idx: int) -> List[Dict[str, Any]]:
        """
        Create chunks from a single corpus document.

        First chunk: title + metadata (author, source, category, published_at)
        Remaining chunks: from body

        Args:
            doc: Document dictionary from corpus
            doc_idx: Document index (used as doc_id)

        Returns:
            List of chunk dictionaries
        """
        doc_id = f"doc_{doc_idx}"
        chunks = []

        # Create title + metadata chunk
        # Convert all fields to strings to ensure JSON serialization
        title = str(doc.get("title", "")) if doc.get("title") else ""
        author = str(doc.get("author", "")) if doc.get("author") else ""
        source = str(doc.get("source", "")) if doc.get("source") else ""
        published_at = str(doc.get("published_at", "")) if doc.get("published_at") else ""
        category = str(doc.get("category", "")) if doc.get("category") else ""
        url = str(doc.get("url", "")) if doc.get("url") else ""

        # Build metadata chunk text
        metadata_parts = [f"Title: {title}"]
        if author:
            metadata_parts.append(f"Author: {author}")
        if source:
            metadata_parts.append(f"Source: {source}")
        if category:
            metadata_parts.append(f"Category: {category}")
        if published_at:
            metadata_parts.append(f"Published: {published_at}")
        if url:
            metadata_parts.append(f"URL: {url}")

        metadata_text = "\n".join(metadata_parts)

        chunks.append({
            "chunk_id": f"{doc_id}_metadata",
            "text": metadata_text,
            "doc_id": doc_id,
            "doc_idx": doc_idx,
            "chunk_index": 0,
            "chunk_type": "metadata",
            "title": title,
            "author": author,
            "source": source,
            "category": category,
            "published_at": published_at,
            "url": url
        })

        # Process body
        body = doc.get("body", "")

        if body:
            # Chunk the body text
            body_chunks = self.chunker.chunk_text(body, doc_id=doc_id)

            # Adjust chunk indices and add metadata
            for chunk in body_chunks:
                # Update chunk_index to account for metadata chunk
                chunk["chunk_index"] = chunk["chunk_index"] + 1
                chunk["chunk_id"] = f"{doc_id}_body_{chunk['chunk_index']}"
                chunk["chunk_type"] = "body"
                chunk["title"] = title
                chunk["doc_idx"] = doc_idx
                chunk["author"] = author
                chunk["source"] = source
                chunk["category"] = category

            chunks.extend(body_chunks)

        return chunks

    def process_corpus_to_chunks(self, corpus: Any) -> List[Dict[str, Any]]:
        """
        Process entire corpus into chunks.

        Args:
            corpus: Corpus dataset

        Returns:
            List of all chunks from all documents
        """
        print("Creating chunks from corpus documents...")
        all_chunks = []

        for doc_idx, doc in enumerate(tqdm(corpus, desc="Processing documents")):
            doc_chunks = self.create_chunks_from_document(doc, doc_idx)
            all_chunks.extend(doc_chunks)

        print(f"Created {len(all_chunks)} total chunks from {len(corpus)} documents")
        return all_chunks

    def save_chunks(self, chunks: List[Dict[str, Any]], filename: str = "chunks.jsonl"):
        """
        Save chunks to JSONL file.

        Args:
            chunks: List of chunk dictionaries
            filename: Output filename
        """
        output_path = self.output_dir / filename
        print(f"Saving chunks to {output_path}...")

        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"Saved {len(chunks)} chunks")

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Build FAISS index from chunks.

        Args:
            chunks: List of chunk dictionaries
        """
        print("Building FAISS index...")

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]

        # Get embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_client.embed_texts(texts, show_progress=True)

        # Create index builder
        embedding_dim = embeddings.shape[1]
        self.index_builder = FAISSIndexBuilder(
            embedding_dim=embedding_dim,
            index_type="flat",
            metric="cosine"
        )

        # Build index
        self.index_builder.build_index(embeddings, chunks)

        # Create retriever
        self.retriever = VectorRetriever(self.index_builder, self.embedding_client)

    def save_index(
        self,
        index_filename: str = "index.faiss",
        mapping_filename: str = "chunk_id_mapping.json"
    ):
        """
        Save FAISS index and chunk ID mapping.

        Args:
            index_filename: Name for FAISS index file
            mapping_filename: Name for chunk ID mapping file
        """
        index_path = str(self.output_dir / index_filename)

        # Save FAISS index and metadata
        self.index_builder.save(index_path)

        # Create chunk_id mapping (index -> chunk_id)
        mapping = {}
        for idx, chunk_meta in enumerate(self.index_builder.metadata):
            mapping[idx] = chunk_meta.get("chunk_id", f"unknown_{idx}")

        # Save mapping
        mapping_path = self.output_dir / mapping_filename
        print(f"Saving chunk ID mapping to {mapping_path}...")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        print(f"Saved mapping with {len(mapping)} entries")

    def extract_queries(self, queries_dataset: Any) -> List[Dict[str, Any]]:
        """
        Extract queries from MultiHop-RAG queries dataset.

        Args:
            queries_dataset: Queries dataset

        Returns:
            List of query dictionaries with metadata
        """
        print("Extracting queries from dataset...")
        queries = []

        for query_idx, query_data in enumerate(tqdm(queries_dataset, desc="Extracting queries")):
            query_text = query_data.get("query", "")

            if not query_text:
                continue

            query_id = f"query_{query_idx}"

            queries.append({
                "query_id": query_id,
                "query_idx": query_idx,
                "query": query_text,
                "answer": query_data.get("answer", ""),
                "question_type": query_data.get("question_type", ""),
                "evidence_list": query_data.get("evidence_list", [])
            })

        print(f"Extracted {len(queries)} queries")
        return queries

    def search_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks for each query and convert to TaoTie format.

        Args:
            queries: List of query dictionaries

        Returns:
            List of query dictionaries in TaoTie format
        """
        if self.retriever is None:
            raise RuntimeError("Index not built. Call build_index first.")

        print(f"Searching for relevant chunks (top_k={self.top_k})...")
        results = []

        for query in tqdm(queries, desc="Searching queries"):
            query_text = query["query"]

            # Search for relevant chunks
            retrieved = self.retriever.search(
                query=query_text,
                top_k=self.top_k,
                return_scores=True
            )

            # Convert to TaoTie format
            contexts = []
            for i, chunk in enumerate(retrieved, 1):
                # Format each chunk as "Passage X:\n{text}"
                passage_text = f"Passage {i}:\n{chunk['text']}"
                contexts.append(passage_text)

            # Calculate total context length
            total_length = sum(len(ctx) for ctx in contexts)

            # Build TaoTie format entry
            taotie_entry = {
                "id": query["query_id"],
                "num_ctxs": len(contexts),
                "length": total_length,
                "question": query["query"],
                "answer": [query["answer"]] if isinstance(query["answer"], str) else query["answer"],
                "contexts": contexts
            }

            results.append(taotie_entry)

        return results

    def save_queries(
        self,
        query_results: List[Dict[str, Any]],
        filename: str = "multihop_rag.jsonl"
    ):
        """
        Save queries in TaoTie format to JSONL file.

        Args:
            query_results: List of query result dictionaries in TaoTie format
            filename: Output filename (default: multihop_rag.jsonl for TaoTie)
        """
        output_path = self.output_dir / filename
        print(f"Saving TaoTie format queries to {output_path}...")

        with open(output_path, "w", encoding="utf-8") as f:
            for result in query_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Saved {len(query_results)} queries in TaoTie format")

    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        print(f"\n{'='*60}")
        print("MultiHop-RAG Evaluation Pipeline")
        print(f"{'='*60}\n")

        # Step 1: Load datasets
        corpus, queries_dataset = self.load_multihop_dataset()

        # Step 2: Create chunks
        chunks = self.process_corpus_to_chunks(corpus)

        # Step 3: Save chunks
        self.save_chunks(chunks)

        # Step 4: Build index
        self.build_index(chunks)

        # Step 5: Save index
        self.save_index()

        # Step 6: Extract queries
        queries = self.extract_queries(queries_dataset)

        # Step 7: Search queries
        query_results = self.search_queries(queries)

        # Step 8: Save query results
        self.save_queries(query_results)

        print(f"\n{'='*60}")
        print("Evaluation complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        # Print summary statistics
        self.print_summary(chunks, query_results)

    def print_summary(self, chunks: List[Dict[str, Any]], query_results: List[Dict[str, Any]]):
        """Print summary statistics."""
        print("\nSummary:")
        print(f"  Total chunks: {len(chunks)}")

        metadata_chunks = sum(1 for c in chunks if c.get("chunk_type") == "metadata")
        body_chunks = sum(1 for c in chunks if c.get("chunk_type") == "body")

        print(f"  - Metadata chunks: {metadata_chunks}")
        print(f"  - Body chunks: {body_chunks}")
        print(f"  Total queries: {len(query_results)}")
        print(f"  Top-k per query: {self.top_k}")
        print("\nOutput files:")
        print(f"  - {self.output_dir}/chunks.jsonl")
        print(f"  - {self.output_dir}/index.faiss")
        print(f"  - {self.output_dir}/index.faiss.meta")
        print(f"  - {self.output_dir}/chunk_id_mapping.json")
        print(f"  - {self.output_dir}/multihop_rag.jsonl (TaoTie format)")
        print("\nTaoTie format includes: id, num_ctxs, length, question, answer, contexts")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MultiHop-RAG dataset evaluation for RAG")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./multihop_rag_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--embedding-api-url",
        type=str,
        default="http://localhost:30000/v1/embeddings",
        help="Embedding API URL"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="gte-Qwen2-7B-Instruct",
        help="Embedding model name"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size in tokens"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap in tokens"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = MultiHopRAGEvaluator(
        output_dir=args.output_dir,
        embedding_api_url=args.embedding_api_url,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k
    )

    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
