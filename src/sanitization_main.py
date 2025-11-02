#!/usr/bin/env python3

import argparse
import logging
import os
import time
import pandas as pd
from sanitization_engine.manager import run_full_pipeline
from sanitization_engine.sanitizer import aggregate_flags, sanitize_data


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Full Data Sanitization Engine")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run entire preprocessing, contamination detection, and membership checking pipeline before sanitization.")
    parser.add_argument("--raw-data-path", type=str, default=None,
                        help="Raw input data for preprocessing.")
    parser.add_argument("--use-default-raw-data", action="store_true",
                        help="Use the default raw data for preprocessing.")
    parser.add_argument("--sanitization-action", choices=["remove", "anonymize", "rewrite"],
                        default="remove", help="Sanitization action to perform.")
    parser.add_argument("--sanitized-output", type=str, default="data/sanitized_dataset.csv",
                        help="Path to save the sanitized dataset.")
    parser.add_argument("--sanitization-log", type=str, default="data/sanitization_log.csv",
                        help="Path to save detailed sanitization logs.")
    args = parser.parse_args()

    start_time = time.perf_counter()

    if args.full_pipeline:
        run_full_pipeline(args)

    logging.info("Starting Data Sanitization step.")

    # Load necessary data
    # preprocessed_path = "data/preprocessed_wikitext103_subset.csv"
    # contamination_path = "data/contamination_flags.csv"
    # membership_path = "data/membership_inference_flags.csv"
    # for testing purposes
    preprocessed_path = "data/preprocessed_wikitext103_subset_3414.csv"
    contamination_path = "data/contamination_flags_3414.csv"
    membership_path = "data/membership_inference_flags_3414.csv"

    df_preprocessed = pd.read_csv(preprocessed_path)
    df_contamination = pd.read_csv(contamination_path)
    df_membership = pd.read_csv(membership_path)

    flagged_indices, flag_reason = aggregate_flags(df_contamination, df_membership)

    logging.info(f"Total flagged segments: {len(flagged_indices)}")

    sanitized_df, log_df = sanitize_data(df_preprocessed, flagged_indices, flag_reason, args.sanitization_action)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.sanitized_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.sanitization_log), exist_ok=True)

    sanitized_df.to_csv(args.sanitized_output, index=False)
    logging.info(f"Sanitized dataset saved: {args.sanitized_output}")

    log_df.to_csv(args.sanitization_log, index=False)
    logging.info(f"Sanitization log saved: {args.sanitization_log}")

    end_time = time.perf_counter()
    logging.info(f"Sanitization pipeline completed in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()
