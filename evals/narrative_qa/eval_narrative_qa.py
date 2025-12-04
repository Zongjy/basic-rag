"""
NarrativeQA Dataset Evaluation Script

This script:
1. Loads the NarrativeQA dataset from HuggingFace (deepmind/narrativeqa)
2. Chunks each document's text field
3. Builds a FAISS index for semantic search across all chunks
4. Searches for relevant chunks using questions
5. Saves output files in TaoTie format:
   - chunks.jsonl: All document chunks
   - index files: FAISS index and chunk_id mapping
   - narrative_qa.jsonl: Questions in TaoTie format (id, question, answer, contexts)
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


class NarrativeQAEvaluator:
    """NarrativeQA dataset evaluator for RAG system."""

    def __init__(
        self,
        output_dir: str = "./narrative_qa_output",
        embedding_api_url: str = "http://localhost:30000/v1/embeddings",
        embedding_model: str = "gte-Qwen2-7B-Instruct",
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        top_k: int = 5,
        split: str = "test"
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
            split: Dataset split to use (train/validation/test)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.top_k = top_k
        self.split = split

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

    def load_narrative_qa_dataset(self) -> Any:
        """
        Load NarrativeQA dataset from HuggingFace.

        Returns:
            Dataset split (contains both documents and questions)
        """
        print(f"Loading NarrativeQA dataset (split={self.split})...")
        dataset = load_dataset("deepmind/narrativeqa", split=self.split)
        print(f"Loaded {len(dataset)} examples")
        return dataset

    def create_chunks_from_document(
        self,
        doc_text: str,
        doc_id: str,
        doc_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from a single document.

        First chunk: document metadata (title, kind, url, etc.)
        Remaining chunks: from document text

        Args:
            doc_text: Full document text
            doc_id: Document ID
            doc_metadata: Document metadata (from document field)

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        # Create metadata chunk
        title = str(doc_metadata.get("title", "")) if doc_metadata.get("title") else ""
        kind = str(doc_metadata.get("kind", "")) if doc_metadata.get("kind") else ""
        url = str(doc_metadata.get("url", "")) if doc_metadata.get("url") else ""
        summary_text = ""
        if doc_metadata.get("summary") and doc_metadata["summary"].get("text"):
            summary_text = str(doc_metadata["summary"]["text"])

        # Build metadata chunk text
        metadata_parts = [f"Title: {title}"]
        if kind:
            metadata_parts.append(f"Type: {kind}")
        if url:
            metadata_parts.append(f"URL: {url}")
        if summary_text:
            metadata_parts.append(f"Summary: {summary_text}")

        metadata_text = "\n".join(metadata_parts)

        chunks.append({
            "chunk_id": f"{doc_id}_metadata",
            "text": metadata_text,
            "doc_id": doc_id,
            "chunk_index": 0,
            "chunk_type": "metadata",
            "title": title,
            "kind": kind,
            "url": url
        })

        # Process document text
        if doc_text:
            # Chunk the document text
            body_chunks = self.chunker.chunk_text(doc_text, doc_id=doc_id)

            # Adjust chunk indices and add metadata
            for chunk in body_chunks:
                # Update chunk_index to account for metadata chunk
                chunk["chunk_index"] = chunk["chunk_index"] + 1
                chunk["chunk_id"] = f"{doc_id}_text_{chunk['chunk_index']}"
                chunk["chunk_type"] = "text"
                chunk["title"] = title

            chunks.extend(body_chunks)

        return chunks

    def process_dataset_to_chunks(self, dataset: Any) -> List[Dict[str, Any]]:
        """
        Process entire dataset into chunks.

        Each example in NarrativeQA has a document with text and a question.
        We create chunks from all unique documents.

        Args:
            dataset: NarrativeQA dataset split

        Returns:
            List of all chunks from all documents
        """
        print("Creating chunks from documents...")
        all_chunks = []
        seen_doc_ids = set()

        for example_idx, example in enumerate(tqdm(dataset, desc="Processing documents")):
            # Get document info
            document = example["document"]
            doc_id = str(document["id"])

            # Skip if we've already processed this document
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            # Get document text
            doc_text = document.get("text", "")

            if not doc_text:
                print(f"Warning: Document {doc_id} has no text, skipping")
                continue

            # Create chunks from this document
            doc_chunks = self.create_chunks_from_document(
                doc_text=doc_text,
                doc_id=doc_id,
                doc_metadata=document
            )
            all_chunks.extend(doc_chunks)

        print(f"Created {len(all_chunks)} total chunks from {len(seen_doc_ids)} unique documents")
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

    def extract_questions(self, dataset: Any) -> List[Dict[str, Any]]:
        """
        Extract questions from NarrativeQA dataset.

        Args:
            dataset: NarrativeQA dataset split

        Returns:
            List of question dictionaries with metadata
        """
        print("Extracting questions from dataset...")
        questions = []

        for example_idx, example in enumerate(tqdm(dataset, desc="Extracting questions")):
            question_data = example.get("question", {})
            question_text = question_data.get("text", "")

            if not question_text:
                continue

            # Get answers (list of answer dicts with text and tokens)
            answers = example.get("answers", [])
            answer_texts = []
            for ans in answers:
                if ans and ans.get("text"):
                    answer_texts.append(ans["text"])

            # Get document ID for reference
            document = example.get("document", {})
            doc_id = str(document.get("id", ""))

            question_id = f"question_{example_idx}"

            questions.append({
                "question_id": question_id,
                "question_idx": example_idx,
                "question": question_text,
                "answers": answer_texts,
                "doc_id": doc_id
            })

        print(f"Extracted {len(questions)} questions")
        return questions

    def search_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks for each question and convert to TaoTie format.

        Args:
            questions: List of question dictionaries

        Returns:
            List of question dictionaries in TaoTie format
        """
        if self.retriever is None:
            raise RuntimeError("Index not built. Call build_index first.")

        print(f"Searching for relevant chunks (top_k={self.top_k})...")
        results = []

        for question in tqdm(questions, desc="Searching questions"):
            question_text = question["question"]

            # Search for relevant chunks
            retrieved = self.retriever.search(
                query=question_text,
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
                "id": question["question_id"],
                "num_ctxs": len(contexts),
                "length": total_length,
                "question": question["question"],
                "answer": question["answers"],
                "contexts": contexts
            }

            results.append(taotie_entry)

        return results

    def save_questions(
        self,
        question_results: List[Dict[str, Any]],
        filename: str = "narrative_qa.jsonl"
    ):
        """
        Save questions in TaoTie format to JSONL file.

        Args:
            question_results: List of question result dictionaries in TaoTie format
            filename: Output filename (default: narrative_qa.jsonl for TaoTie)
        """
        output_path = self.output_dir / filename
        print(f"Saving TaoTie format questions to {output_path}...")

        with open(output_path, "w", encoding="utf-8") as f:
            for result in question_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Saved {len(question_results)} questions in TaoTie format")

    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        print(f"\n{'='*60}")
        print("NarrativeQA Evaluation Pipeline")
        print(f"{'='*60}\n")

        # Step 1: Load dataset
        dataset = self.load_narrative_qa_dataset()

        # Step 2: Create chunks
        chunks = self.process_dataset_to_chunks(dataset)

        # Step 3: Save chunks
        self.save_chunks(chunks)

        # Step 4: Build index
        self.build_index(chunks)

        # Step 5: Save index
        self.save_index()

        # Step 6: Extract questions
        questions = self.extract_questions(dataset)

        # Step 7: Search questions
        question_results = self.search_questions(questions)

        # Step 8: Save question results
        self.save_questions(question_results)

        print(f"\n{'='*60}")
        print("Evaluation complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        # Print summary statistics
        self.print_summary(chunks, question_results)

    def print_summary(self, chunks: List[Dict[str, Any]], question_results: List[Dict[str, Any]]):
        """Print summary statistics."""
        print("\nSummary:")
        print(f"  Total chunks: {len(chunks)}")

        metadata_chunks = sum(1 for c in chunks if c.get("chunk_type") == "metadata")
        text_chunks = sum(1 for c in chunks if c.get("chunk_type") == "text")

        print(f"  - Metadata chunks: {metadata_chunks}")
        print(f"  - Text chunks: {text_chunks}")
        print(f"  Total questions: {len(question_results)}")
        print(f"  Top-k per question: {self.top_k}")
        print("\nOutput files:")
        print(f"  - {self.output_dir}/chunks.jsonl")
        print(f"  - {self.output_dir}/index.faiss")
        print(f"  - {self.output_dir}/index.faiss.meta")
        print(f"  - {self.output_dir}/chunk_id_mapping.json")
        print(f"  - {self.output_dir}/narrative_qa.jsonl (TaoTie format)")
        print("\nTaoTie format includes: id, num_ctxs, length, question, answer, contexts")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="NarrativeQA dataset evaluation for RAG")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./narrative_qa_output",
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
        help="Number of chunks to retrieve per question"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = NarrativeQAEvaluator(
        output_dir=args.output_dir,
        embedding_api_url=args.embedding_api_url,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        split=args.split
    )

    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
