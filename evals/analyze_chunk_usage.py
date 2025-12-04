"""
Chunk Usage Statistics and CDF Plotting

This script analyzes chunk retrieval patterns from RAG evaluation results
and generates a Cumulative Distribution Function (CDF) plot to identify
hot spot data (frequently retrieved chunks).

Usage:
    python analyze_chunk_usage.py --dataset multihop-rag --output-dir ./multihop_rag_1024_15
    python analyze_chunk_usage.py --dataset narrative-qa --output-dir ./narrative_qa_output
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Counter as CounterType
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class ChunkUsageAnalyzer:
    """Analyze chunk usage patterns from RAG evaluation results."""

    def __init__(self, dataset: str, output_dir: str):
        """
        Initialize analyzer.

        Args:
            dataset: Dataset name ('multihop-rag' or 'narrative-qa')
            output_dir: Directory containing evaluation results
        """
        self.dataset = dataset
        self.output_dir = Path(output_dir)

        # Determine result file name based on dataset
        if dataset == "multihop-rag":
            self.result_file = "multihop_rag.jsonl"
        elif dataset == "narrative-qa":
            self.result_file = "narrative_qa.jsonl"
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Use 'multihop-rag' or 'narrative-qa'")

        self.result_path = self.output_dir / self.result_file
        self.chunks_path = self.output_dir / "chunks.jsonl"

        # Validate paths
        if not self.result_path.exists():
            raise FileNotFoundError(f"Result file not found: {self.result_path}")
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")

    def extract_chunk_ids_from_contexts(self, contexts: List[str]) -> List[str]:
        """
        Extract chunk IDs from context passages.

        Since contexts are formatted as "Passage X:\n{text}", we need to
        match them against the original chunks to find chunk IDs.

        Args:
            contexts: List of context strings from TaoTie format

        Returns:
            List of chunk texts for matching
        """
        chunk_texts = []

        for context in contexts:
            # Remove "Passage X:\n" prefix
            if context.startswith("Passage "):
                lines = context.split("\n", 1)
                if len(lines) > 1:
                    text = lines[1]
                else:
                    text = context
            else:
                text = context

            chunk_texts.append(text)

        return chunk_texts

    def load_chunks_mapping(self) -> Dict[str, str]:
        """
        Load chunks and create text->chunk_id mapping.

        Returns:
            Dictionary mapping chunk text to chunk_id
        """
        print(f"Loading chunks from {self.chunks_path}...")
        text_to_id = {}

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Building chunk mapping"):
                chunk = json.loads(line)
                chunk_id = chunk.get("chunk_id")
                text = chunk.get("text", "")

                if chunk_id and text:
                    text_to_id[text] = chunk_id

        print(f"Loaded {len(text_to_id)} unique chunks")
        return text_to_id

    def analyze_chunk_usage(self) -> CounterType[str]:
        """
        Analyze chunk usage from evaluation results.

        Returns:
            Counter object mapping chunk_id to usage count
        """
        print(f"Analyzing chunk usage from {self.result_path}...")

        # Load chunk text->id mapping
        text_to_id = self.load_chunks_mapping()

        # Count chunk usage
        chunk_usage = Counter()
        total_queries = 0
        matched_chunks = 0
        unmatched_chunks = 0

        with open(self.result_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Processing queries"):
                result = json.loads(line)
                total_queries += 1

                contexts = result.get("contexts", [])

                # Extract chunk texts
                chunk_texts = self.extract_chunk_ids_from_contexts(contexts)

                # Map texts to chunk IDs
                for text in chunk_texts:
                    chunk_id = text_to_id.get(text)
                    if chunk_id:
                        chunk_usage[chunk_id] += 1
                        matched_chunks += 1
                    else:
                        unmatched_chunks += 1

        print(f"\nStatistics:")
        print(f"  Total queries: {total_queries}")
        print(f"  Total chunk retrievals: {matched_chunks + unmatched_chunks}")
        print(f"  Matched chunks: {matched_chunks}")
        print(f"  Unmatched chunks: {unmatched_chunks}")
        print(f"  Unique chunks used: {len(chunk_usage)}")
        print(f"  Total chunks available: {len(text_to_id)}")
        print(f"  Chunk usage rate: {len(chunk_usage) / len(text_to_id) * 100:.2f}%")

        return chunk_usage

    def compute_statistics(self, chunk_usage: CounterType[str], total_chunks: int) -> Dict[str, Any]:
        """
        Compute usage statistics.

        Args:
            chunk_usage: Counter of chunk usage
            total_chunks: Total number of chunks in corpus

        Returns:
            Dictionary of statistics
        """
        usage_counts = list(chunk_usage.values())

        # Add zero counts for unused chunks
        num_unused = total_chunks - len(chunk_usage)
        all_counts = usage_counts + [0] * num_unused

        stats = {
            "total_chunks": total_chunks,
            "used_chunks": len(chunk_usage),
            "unused_chunks": num_unused,
            "total_retrievals": sum(usage_counts),
            "mean_usage": np.mean(all_counts),
            "median_usage": np.median(all_counts),
            "std_usage": np.std(all_counts),
            "max_usage": max(all_counts) if all_counts else 0,
            "min_usage": min(all_counts) if all_counts else 0,
        }

        # Percentiles
        for p in [50, 75, 90, 95, 99]:
            stats[f"p{p}_usage"] = np.percentile(all_counts, p)

        # Top-K most used chunks
        top_10 = chunk_usage.most_common(10)
        stats["top_10_chunks"] = top_10

        return stats

    def plot_cdf(
        self,
        chunk_usage: CounterType[str],
        total_chunks: int,
        output_path: Path,
        title_suffix: str = ""
    ):
        """
        Plot Cumulative Distribution Function of chunk usage.

        Args:
            chunk_usage: Counter of chunk usage
            total_chunks: Total number of chunks in corpus
            output_path: Path to save the plot
            title_suffix: Additional text for plot title
        """
        print(f"Generating CDF plot...")

        # Get usage counts
        usage_counts = list(chunk_usage.values())

        # Add zero counts for unused chunks
        num_unused = total_chunks - len(chunk_usage)
        all_counts = usage_counts + [0] * num_unused

        # Sort counts
        sorted_counts = np.sort(all_counts)

        # Compute CDF
        n = len(sorted_counts)
        cdf = np.arange(1, n + 1) / n

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot CDF
        plt.subplot(1, 2, 1)
        plt.plot(sorted_counts, cdf, linewidth=2)
        plt.xlabel("Number of Times Retrieved", fontsize=12)
        plt.ylabel("Cumulative Probability", fontsize=12)
        plt.title(f"CDF of Chunk Usage{title_suffix}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)

        # Add statistics text
        unused_pct = num_unused / total_chunks * 100
        plt.axhline(y=unused_pct/100, color='r', linestyle='--', alpha=0.5, label=f'Unused: {unused_pct:.1f}%')
        plt.legend()

        # Plot histogram (log scale)
        plt.subplot(1, 2, 2)

        # Remove zeros for histogram
        non_zero_counts = [c for c in all_counts if c > 0]

        if non_zero_counts:
            plt.hist(non_zero_counts, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel("Number of Times Retrieved", fontsize=12)
            plt.ylabel("Number of Chunks", fontsize=12)
            plt.title(f"Distribution of Chunk Usage (Non-zero){title_suffix}", fontsize=14, fontweight="bold")
            plt.yscale('log')
            plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved CDF plot to {output_path}")

        # Also create a separate log-log plot for hot spot analysis
        plt.figure(figsize=(10, 6))

        # Sort chunks by usage (descending)
        sorted_usage = sorted(chunk_usage.values(), reverse=True)
        ranks = np.arange(1, len(sorted_usage) + 1)

        plt.loglog(ranks, sorted_usage, linewidth=2, marker='o', markersize=3, alpha=0.7)
        plt.xlabel("Chunk Rank (by usage)", fontsize=12)
        plt.ylabel("Number of Times Retrieved", fontsize=12)
        plt.title(f"Chunk Usage Hot Spot Analysis{title_suffix}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3, which="both")

        # Add annotations for top chunks
        for i in [0, 9, 99] if len(sorted_usage) > 100 else [0]:
            if i < len(sorted_usage):
                plt.annotate(
                    f"Top {i+1}: {sorted_usage[i]}",
                    xy=(ranks[i], sorted_usage[i]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )

        plt.tight_layout()

        # Save hot spot plot
        hotspot_path = output_path.parent / f"{output_path.stem}_hotspot.png"
        plt.savefig(hotspot_path, dpi=300, bbox_inches='tight')
        print(f"Saved hot spot plot to {hotspot_path}")

        plt.close('all')

    def print_statistics(self, stats: Dict[str, Any]):
        """Print statistics summary."""
        print("\n" + "="*60)
        print("Chunk Usage Statistics")
        print("="*60)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Used chunks: {stats['used_chunks']} ({stats['used_chunks']/stats['total_chunks']*100:.2f}%)")
        print(f"Unused chunks: {stats['unused_chunks']} ({stats['unused_chunks']/stats['total_chunks']*100:.2f}%)")
        print(f"\nTotal retrievals: {stats['total_retrievals']}")
        print(f"Mean usage per chunk: {stats['mean_usage']:.2f}")
        print(f"Median usage: {stats['median_usage']:.2f}")
        print(f"Std deviation: {stats['std_usage']:.2f}")
        print(f"Max usage: {stats['max_usage']}")

        print(f"\nPercentiles:")
        for p in [50, 75, 90, 95, 99]:
            print(f"  P{p}: {stats[f'p{p}_usage']:.2f}")

        print(f"\nTop 10 Most Retrieved Chunks:")
        for i, (chunk_id, count) in enumerate(stats['top_10_chunks'], 1):
            pct = count / stats['total_retrievals'] * 100
            print(f"  {i}. {chunk_id}: {count} times ({pct:.2f}%)")

        print("="*60 + "\n")

    def run_analysis(self):
        """Run complete analysis pipeline."""
        print(f"\n{'='*60}")
        print(f"Chunk Usage Analysis: {self.dataset}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        # Analyze chunk usage
        chunk_usage = self.analyze_chunk_usage()

        # Get total chunks count
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            total_chunks = sum(1 for _ in f)

        # Compute statistics
        stats = self.compute_statistics(chunk_usage, total_chunks)

        # Print statistics
        self.print_statistics(stats)

        # Plot CDF
        plot_path = self.output_dir / "chunk_usage_cdf.png"
        self.plot_cdf(chunk_usage, total_chunks, plot_path, f" - {self.dataset}")

        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"Output files:")
        print(f"  - {plot_path}")
        print(f"  - {plot_path.parent / (plot_path.stem + '_hotspot.png')}")
        print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze chunk usage patterns and generate CDF plots from RAG evaluation results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["multihop-rag", "narrative-qa"],
        help="Dataset name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing evaluation results (with .jsonl files)"
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = ChunkUsageAnalyzer(
        dataset=args.dataset,
        output_dir=args.output_dir
    )

    # Run analysis
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
