#!/usr/bin/env python3
"""
Membership Inference Checker Module

This script performs neighborhood-based membership inference by:
  1. Loading preprocessed data (or generating it via the preprocessor module if missing).
  2. Computing or loading segment embeddings.
  3. Running a nearest neighbor search to compute maximum cosine similarity for each segment.
  4. Flagging segments as duplicates (if similarity ≥ high threshold) or outliers (if similarity < low threshold).
  5. Saving the results to a CSV file.
  6. Generating and saving visualizations to the directory data/plots/membership_module_plots.

Usage:
    python main.py [--input-file PATH] [--embeddings-file PATH] [--output-file PATH] [other options...]
"""

import os
import argparse
import logging
import time
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

# Set aesthetic style for plots
sns.set(style="whitegrid", palette="viridis", font_scale=1.2)

from embeddings import load_preprocessed_data, compute_embeddings_for_segments
from neighborhood import flag_membership


def process_membership_inference(args):
    # Load preprocessed data; if not available, run the preprocessor module
    df = load_preprocessed_data(args.input_file, preprocess_if_missing=True)
    logging.info("Loaded data with shape: %s", df.shape)

    # Ensure 'segments' column is string type
    df['segments'] = df['segments'].astype(str)

    # Add stable segment identifiers for downstream merging if absent
    if 'segment_id' not in df.columns:
        df['segment_id'] = df['segments'].map(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()[:16])

    # Compute or load embeddings
    embeddings = compute_embeddings_for_segments(
        df,
        text_column='segments',
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        embeddings_file=args.embeddings_file
    )

    # Compute neighborhood similarity and flag membership issues
    logging.info("Computing neighborhood similarity and flagging membership issues...")
    duplicate_flags, outlier_flags, max_neighbor_sim = flag_membership(
        embeddings,
        high_sim_threshold=args.high_sim_threshold,
        low_sim_threshold=args.low_sim_threshold,
        n_neighbors=args.n_neighbors
    )

    df['max_neighbor_similarity'] = max_neighbor_sim
    df['duplicate_flag'] = duplicate_flags
    df['outlier_flag'] = outlier_flags
    df['membership_inference_flag'] = duplicate_flags | outlier_flags

    return df


def save_plots(df, high_sim_threshold, low_sim_threshold, output_plots_dir):
    os.makedirs(output_plots_dir, exist_ok=True)

    # Plot 1: Distribution of Max Neighbor Cosine Similarity
    plt.figure(figsize=(10, 6))
    sns.histplot(df['max_neighbor_similarity'], bins=50, kde=True, color="steelblue")
    plt.axvline(x=high_sim_threshold, color='red', linestyle='--', label=f'High Threshold ({high_sim_threshold})')
    plt.axvline(x=low_sim_threshold, color='orange', linestyle='--', label=f'Low Threshold ({low_sim_threshold})')
    plt.xlabel("Max Neighbor Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Distribution of Max Neighbor Cosine Similarity")
    plt.legend()
    plot1_path = os.path.join(output_plots_dir, "max_neighbor_similarity_hist.png")
    plt.tight_layout()
    plt.savefig(plot1_path)
    plt.close()
    logging.info("Saved plot: %s", plot1_path)

    # Plot 2: Bar Plot of Flagged Categories
    flag_counts = {
        "Duplicates": int(df['duplicate_flag'].sum()),
        "Outliers": int(df['outlier_flag'].sum()),
        "Non-Flagged": int(len(df) - df['membership_inference_flag'].sum())
    }
    flag_counts_df = pd.DataFrame(list(flag_counts.items()), columns=['Flag Category', 'Count'])

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Flag Category', y='Count', data=flag_counts_df, palette="magma")
    plt.title("Count of Segments by Flag Category")
    plt.xlabel("Flag Category")
    plt.ylabel("Number of Segments")
    plt.tight_layout()
    plot2_path = os.path.join(output_plots_dir, "flagged_categories_bar.png")
    plt.savefig(plot2_path)
    plt.close()
    logging.info("Saved plot: %s", plot2_path)

    # Plot 3: Scatter Plot of Max Neighbor Similarity Across Segments
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df.index, y=df['max_neighbor_similarity'], hue=df['membership_inference_flag'],
                    palette="coolwarm", legend="full")
    plt.xlabel("Segment Index")
    plt.ylabel("Max Neighbor Cosine Similarity")
    plt.title("Variation of Max Neighbor Similarity Across Segments")
    plt.legend(title="Flagged")
    plt.tight_layout()
    plot3_path = os.path.join(output_plots_dir, "neighbor_similarity_scatter.png")
    plt.savefig(plot3_path)
    plt.close()
    logging.info("Saved plot: %s", plot3_path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Membership Inference Checker Module")
    parser.add_argument("--input-file", type=str, default="data/preprocessed_wikitext103_subset.csv",
                        help="Path to preprocessed CSV file with a 'segments' column.")
    parser.add_argument("--embeddings-file", type=str, default="data/segment_embeddings.npy",
                        help="Path to load/save segment embeddings.")
    parser.add_argument("--output-file", type=str, default="data/membership_inference_flags.csv",
                        help="Path to save the membership inference results CSV.")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model for computing embeddings.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding computation.")
    parser.add_argument("--n-neighbors", type=int, default=6,
                        help="Number of nearest neighbors to consider (default: 6).")
    parser.add_argument("--high-sim-threshold", type=float, default=0.95,
                        help="Threshold for high similarity to flag duplicates (default: 0.95).")
    parser.add_argument("--low-sim-threshold", type=float, default=0.3,
                        help="Threshold for low similarity to flag outliers (default: 0.3).")
    parser.add_argument("--plots-dir", type=str, default="results/plots/membership_module_plots",
                        help="Directory to save membership inference plots.")
    args = parser.parse_args()

    logging.info("Starting Membership Inference Checker Module...")
    start_time = time.perf_counter()
    df_result = process_membership_inference(args)
    end_time = time.perf_counter()

    logging.info("Membership inference processing completed in %.2f seconds", end_time - start_time)
    logging.info("Total flagged segments: %d", df_result['membership_inference_flag'].sum())

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df_result.to_csv(args.output_file, index=False)
    logging.info("Membership inference results saved to: %s", args.output_file)

    # Save plots to the specified directory
    save_plots(df_result, args.high_sim_threshold, args.low_sim_threshold, args.plots_dir)


if __name__ == "__main__":
    main()
