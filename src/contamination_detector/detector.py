#!/usr/bin/env python3
"""
Contamination Detector Module

This script integrates:
    - Reference Benchmark Comparison
    - PaCoST-Inspired Confidence Testing

It processes a preprocessed CSV file (which should have a 'segments' column),
computes reference similarity and perplexity-based confidence scores for each segment,
and flags segments that appear contaminated.

Usage:
    python detector.py [--input-file PATH] [--output-file PATH] [other options...]
"""

import os
import argparse
import logging
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from reference_comparison import load_reference_data, check_reference_similarity
from pacost import perturb_text, load_language_model, compute_perplexity


def detect_contamination(args):
    logging.info("Loading preprocessed data from: %s", args.input_file)
    df = pd.read_csv(args.input_file, on_bad_lines='skip', engine='python')
    logging.info("Loaded data shape: %s", df.shape)

    if 'segments' not in df.columns:
        logging.error("Input data must have a 'segments' column.")
        return None

    # Ensure stable identifier for segments
    import hashlib
    df['segments'] = df['segments'].astype(str)
    if 'segment_id' not in df.columns:
        df['segment_id'] = df['segments'].map(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()[:16])

    # Setup reference benchmark
    logging.info("Setting up reference benchmark comparison...")
    if args.reference_file:
        with open(args.reference_file, 'r') as f:
            reference_texts = [line.strip() for line in f if line.strip()]
    else:
        # reference_texts = [
        #     "this is a known contaminated text from benchmark dataset",
        #     "another reference text that should not be in the training data",
        #     "benchmark evaluation text that must remain separate"
        # ]
        pg19_passages = load_dataset("deepmind/pg19", split="train", num_proc=10, trust_remote_code=True)
        reference_texts = pg19_passages["text"]

    ref_model, ref_embeddings = load_reference_data(reference_texts, model_name=args.ref_model_name)

    # Setup language model for confidence testing
    logging.info("Loading language model for confidence testing...")
    lm_model, lm_tokenizer = load_language_model(model_name=args.lm_model_name)

    ref_similarities = []
    ref_flags = []
    perplexity_orig_list = []
    perplexity_perturbed_list = []
    confidence_flags = []

    perplexity_ratio_threshold = args.perplexity_ratio_threshold
    segments = df['segments'].tolist()

    logging.info("Starting contamination detection on %d segments", len(segments))
    for seg in tqdm(segments, desc="Processing segments"):
        # Reference similarity check
        try:
            max_sim, ref_flag = check_reference_similarity(seg, ref_embeddings, threshold=args.ref_similarity_threshold,
                                                           ref_model=ref_model)
        except Exception as e:
            logging.error("Error in reference similarity check: %s", e)
            max_sim, ref_flag = 0.0, False
        ref_similarities.append(max_sim)
        ref_flags.append(ref_flag)

        # Compute perplexity for original segment
        ppl_orig = compute_perplexity(seg, lm_model, lm_tokenizer)
        # Compute perplexity for perturbed segment
        perturbed_seg = perturb_text(seg)
        ppl_perturbed = compute_perplexity(perturbed_seg, lm_model, lm_tokenizer)

        perplexity_orig_list.append(ppl_orig)
        perplexity_perturbed_list.append(ppl_perturbed)

        # Confidence testing: flag if original perplexity is significantly lower
        if (ppl_orig is not None and ppl_perturbed is not None and ppl_perturbed > 0):
            conf_flag = ppl_orig < perplexity_ratio_threshold * ppl_perturbed
        else:
            conf_flag = False
        confidence_flags.append(conf_flag)

    # Combine flags: flag if either reference or confidence flag is true
    combined_flags = [r or c for r, c in zip(ref_flags, confidence_flags)]

    df['ref_similarity'] = ref_similarities
    df['ref_flag'] = ref_flags
    df['ppl_original'] = perplexity_orig_list
    df['ppl_perturbed'] = perplexity_perturbed_list
    df['confidence_flag'] = confidence_flags
    df['contamination_flag'] = combined_flags

    return df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Contamination Detector Module")
    parser.add_argument("--input-file", type=str, default="data/preprocessed_wikitext103_subset.csv",
                        help="Path to preprocessed CSV file with a 'segments' column.")
    parser.add_argument("--output-file", type=str, default="data/contamination_flags_sample.csv",
                        help="Path to save the flagged contamination output CSV.")
    parser.add_argument("--reference-file", type=str, default=None,
                        help="Optional path to a reference benchmark text file (one text per line).")
    parser.add_argument("--ref_similarity_threshold", type=float, default=0.9,
                        help="Threshold for reference similarity (default: 0.9).")
    parser.add_argument("--perplexity_ratio_threshold", type=float, default=0.8,
                        help="Threshold for perplexity ratio (default: 0.8).")
    parser.add_argument("--ref_model_name", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name for reference comparisons.")
    parser.add_argument("--lm_model_name", type=str, default="distilgpt2",
                        help="Lightweight LM model name for computing perplexity.")
    args = parser.parse_args()

    logging.info("Starting Contamination Detector Module...")
    df_result = detect_contamination(args)
    if df_result is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        df_result.to_csv(args.output_file, index=False)
        logging.info("Contamination detection results saved to: %s", args.output_file)


if __name__ == "__main__":
    main()
