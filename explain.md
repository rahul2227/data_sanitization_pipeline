# Data Sanitization Codebase Overview

## Project Intent & High-Level Flow
- The repository implements an end-to-end text data hygiene pipeline that prepares corpora for language-model training. The canonical flow is: preprocess raw text → detect contamination → flag potential memorization risks → apply configurable sanitization actions.
- Each stage produces CSV artifacts under `data/` that the downstream steps consume, and they share stable `segment_id` hashes to cross-reference rows across modules.
- A Streamlit UI mirrors the CLI experience so the pipeline can be explored interactively without running heavyweight models.

## Core Execution Paths
- `src/sanitization_main.py:12` is the orchestrator. It parses CLI arguments, optionally runs the full preprocessing/analysis stack through `run_full_pipeline`, aggregates all contamination and membership flags, and writes sanitized outputs plus a log.
- `src/sanitization_engine/manager.py:5` wraps the command-line modules via `subprocess.run`, chaining the preprocessor, contamination detector, and membership checker with consistent filenames.
- `streamlit_app.py:1` exposes a five-step UI (Data → Preprocess → Contamination → Membership → Sanitize) with cached helpers that reuse the same normalization, segmentation, and sanitization utilities used by the CLI.

## Module Breakdown

### Preprocessor (`src/preprocessor`)
- `preprocessor_main.py:49` caps large datasets by byte budget, normalizes text, tokenizes with Hugging Face models, deduplicates, segments, and emits `segment_id` hashes for reproducible joins.
- `cleaning.py:1` performs HTML stripping, Unicode normalization, lowercasing, optional stopword removal (using NLTK stopwords downloaded at import), and ASCII cleanup.
- `tokenization.py:1` loads `distilgpt2` via `AutoTokenizer` and truncates token lists to 1,024 tokens to keep processing tractable.
- `segmentation.py:1` supports sentence tokenization (NLTK), fixed-length token chunks, or no segmentation; `segment_dataframe` can cap the number of produced rows early for efficiency.
- `deduplication.py:1` removes exact duplicates, while `contamination_simulator.py:1` can inject synthetic noise using Gutenberg passages to stress downstream detectors.

### Contamination Detector (`src/contamination_detector`)
- `detector.py:28` reads the preprocessed segments, ensures each row has a `segment_id`, and runs two tests per segment: cosine similarity against a reference corpus and a PaCoST-style perplexity check between original and perturbed text.
- `reference_comparison.py:1` loads a SentenceTransformer model (default `all-MiniLM-L6-v2`) and precomputes embeddings for the reference passages, then scores segments against them.
- `pacost.py:1` loads a lightweight causal LM (default `distilgpt2`) to compute perplexity before and after a simple word shuffling perturbation; the ratio determines `confidence_flag`s.
- Outputs contain both raw metrics (`ref_similarity`, `ppl_original`, etc.) and a boolean `contamination_flag` that is later consumed by the sanitization step.

### Membership Inference Checker (`src/membership_inference_checker`)
- `main.py:34` loads (or generates) the preprocessed segments, materializes embeddings, and labels each segment as duplicated or outlier based on nearest-neighbor cosine similarity statistics.
- `embeddings.py:8` encapsulates CSV loading with an optional backstop that calls the preprocessor if the file is missing, and computes or caches embeddings with SentenceTransformers.
- `neighborhood.py:5` uses scikit-learn `NearestNeighbors` with cosine distance to compute each segment's maximum neighbor similarity and derive duplicate/outlier flags.
- Detailed plots (histogram, bar, scatter) are written to `results/plots/membership_module_plots/` to visualize the similarity distribution versus thresholds.

### Sanitization Engine (`src/sanitization_engine`)
- `sanitizer.py:6` merges contamination and membership flags by `segment_id` when available, falling back to row indices; `sanitize_data` applies `remove`, `anonymize`, or `rewrite` actions and logs each change with the original text.
- `manager.py:14` centralizes the shell-out flow so both CLI and UI runs stay consistent in the file names and arguments they pass to subordinate modules.

### Streamlit Application (`streamlit_app.py`)
- Provides cached preprocessing, TF-IDF based contamination heuristics for quick demos, and TF-IDF or SentenceTransformer embeddings for membership analysis.
- Uses the same `aggregate_flags`/`sanitize_data` pair as the CLI and exposes download buttons for the sanitized dataset and action log.
- Falls back to text tables if Streamlit cannot render DataFrames (e.g., due to PyArrow issues), improving robustness in constrained environments.

## Supporting Assets & Artifacts
- `data/` holds generated CSVs (e.g., `preprocessed_wikitext103_subset.csv` plus smaller test fixtures) that persist across runs.
- `results/` is reserved for analytic artifacts like membership similarity plots; additional subdirectories are created on demand.
- `Documentation/` contains the authored project plan (`approach.md`) and diagrams that summarize the pipeline architecture.
- `Exploratory notebooks/` capture the initial experimentation notebooks for each module; they complement the productionized code.
- `Research Papers/` stores reference PDFs (PaCoST and neighborhood-based MIA) that inspired the contamination and membership strategies.
- `validation module/` currently houses a `data_overlap.ipynb` notebook for manual overlap analysis.

## Data Contracts & Identifiers
- Every pipeline stage expects a `segments` column and, when feasible, a `segment_id` hash (SHA-256 prefix) so contamination and membership signals can be joined reliably without recomputing text features.
- Contamination outputs surface both reference similarity and perplexity metrics, enabling post-hoc threshold tuning.
- Membership outputs include `max_neighbor_similarity`, `duplicate_flag`, and `outlier_flag` to distinguish the nature of the risk before sanitization.
- Sanitization logs capture the index, applied action, reason (`contamination`, `membership`, or both), and original segment text for auditability.

## Dependencies & Tooling
- `environment.yml` provisions the Conda environment with NLP and visualization stacks: Hugging Face `datasets`, `transformers`, `sentence-transformers`, PyTorch, scikit-learn, NLTK, Streamlit, and plotting libraries.
- The pipeline expects internet access the first time it downloads Hugging Face datasets/models (wikitext-103, PG-19, Gutenberg, MiniLM, distilgpt2). Subsequent executions reuse the cached artifacts.
- NLTK stopwords are fetched at import time; ensure the environment can write to the NLTK data directory.

## Operational Notes & Caveats
- The default CLI configuration uses reduced subset sizes (`3414` segments) for faster iteration; adjust `--segment-limit` and filenames when running at full scale.
- Full pipeline runs can take significant time and memory because they sequentially invoke Hugging Face data loading, SentenceTransformer encoding, and torch-based perplexity scoring.
- The Streamlit “contamination lite” path intentionally replaces the heavy embedding flow with TF-IDF, trading fidelity for responsiveness; use the CLI module for production-quality contamination checks.
- Many steps assume ASCII-friendly text after normalization. If multilingual support is required, relax the cleaning stage and revisit tokenizer/model choices.

