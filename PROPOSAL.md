# Data Sanitization — Improvement and Streamlit Hosting Proposal

This document outlines targeted improvements to strengthen the codebase, performance, and developer experience; proposes a pragmatic Streamlit UI to run and visualize the pipeline; and enumerates current issues and gaps to address before broader adoption.

## Executive Summary
- The project is well‑structured into four modules: Preprocessor, Contamination Detector, Membership Inference Checker, and Sanitization Engine. Each module is runnable as a script and has clear responsibilities.
- To make the system robust, reproducible, and user‑friendly, we should: close dependency gaps (torch, sentence‑transformers, scikit‑learn), standardize paths/configs, replace fragile index‑based joins with stable segment IDs, improve logging and I/O, and add smoke tests.
- For hosting on Streamlit, we will reuse existing module functions, add lightweight wrappers that avoid file I/O where possible, and cache models/embeddings. A multi‑step app can preprocess, detect, inspect, and sanitize with progress indicators and downloads.

---

## Current Architecture
- Preprocessor (`src/preprocessor`)
  - Loads dataset (default: Hugging Face wikitext‑103 raw), normalizes, tokenizes, deduplicates, segments, and writes CSV.
  - Entrypoint: `preprocessor_main.py`, core functions are already reusable.
- Contamination Detector (`src/contamination_detector`)
  - Combines reference similarity (SentenceTransformers) and PaCoST‑style perplexity testing (distilgpt2 via Transformers) to flag contamination.
  - Entrypoint: `detector.py`; core logic available via `detect_contamination(args)`.
- Membership Inference Checker (`src/membership_inference_checker`)
  - Embeddings via SentenceTransformers; KNN cosine similarity with scikit‑learn; flags duplicates/outliers; generates plots.
  - Entrypoint: `main.py`; core logic via `process_membership_inference(args)` and helpers.
- Sanitization Engine (`src/sanitization_engine`)
  - Orchestrates full pipeline by subprocess; aggregates flags; performs removal/anonymize/rewrite; saves sanitized dataset and logs.

---

## Known Issues and Gaps
1. Dependencies and environment
   - Missing packages required by code: `torch` (used in PaCoST), `sentence-transformers`, `scikit-learn`, and `streamlit`.
   - Optional but recommended: `faiss-cpu` for scalable nearest neighbors; `rich` for nicer CLI; `typer` for CLI.
   - NLTK data: `punkt` is required for sentence segmentation but not downloaded; only `stopwords` is fetched.

2. Data flow and reproducibility
   - Index‑based aggregation: `sanitize_data` relies on DataFrame indices from contamination and membership CSVs matching the preprocessed DataFrame. This is fragile across saves/loads and any row reordering.
   - No stable ID: segments lack a persistent identifier; makes joins brittle and auditing harder.
   - Default file paths are inconsistent (`data/` vs `../data/`) and relative to varying working directories.

3. Performance and UX
   - Heavy defaults: loading full reference datasets (e.g., PG19) and contamination simulation increases runtime and network I/O.
   - Subprocess pipeline logs can “freeze” the CLI (captured output only shown after completion). No live progress from child processes.
   - Tokenizer/model downloads happen at runtime; no warm‑up or offline cache guidance.

4. API and packaging
   - Modules are script‑oriented; several functions require an `argparse` args object. Thin function wrappers that accept Python parameters (and return DataFrames) will simplify reuse in a Streamlit app.
   - No package metadata (`pyproject.toml`) or CLI entrypoint; imports inside modules use local (non‑package) imports that make execution context‑sensitive.

5. Documentation and structure
   - Root README tree has a mismatch for `sanitization_engine` name; some folder names contain spaces (`validation module/`), which can cause tooling friction.
   - Preprocessor README mentions fuzzy/near‑duplicate logic, but code currently does exact duplicate removal.

6. Smaller correctness polish
   - `preprocessor_main.py` has an unused import: `from pygments.lexer import default`.
   - Defaults hard‑code a test subset size (3414) in multiple places; should be parameterized consistently.

---

## Proposed Improvements (Incremental)

1) Environment and dependencies
- Add to environment (Conda or pip):
  - Core: `torch`, `sentence-transformers`, `scikit-learn`, `streamlit`
  - Optional: `faiss-cpu` (fast KNN), `typer` (CLI), `rich` (logging/CLI), `python-dotenv` (config)
- Ensure NLTK data availability:
  - On first run: `nltk.download("punkt")` (and guard it so it’s idempotent)
- Provide an “offline” note in README for model/dataset caching.

2) Stable identifiers and joins
- Add a stable `segment_id` column everywhere segments exist: `segment_id = sha256(segments).hexdigest()[:16]`.
- Aggregate flags by `segment_id` rather than index; merge on `segment_id` to align contamination + membership + preprocessed data deterministically.

3) Path and config hygiene
- Introduce a small `src/common/paths.py` (or a config module) to compute project‑root‑relative paths via `pathlib`.
- Unify defaults under `data/` and `results/` at the repository root; drop `../data/` paths.
- Add a YAML (`config.yaml`) for thresholds, filenames, and toggles (e.g., use_default_raw_data, simulate_contamination).

4) Streaming logs and progress
- Replace `subprocess.run(capture_output=True)` with streaming (`Popen` + iterating over stdout) or run modules in‑process via function calls where feasible; log to both console and rotating file handlers.
- In Streamlit, surface module‑level progress with `tqdm` progress bars connected via callbacks, or by incremental status updates.

5) Lighter defaults and guardrails
- Make contamination simulation opt‑in (default False); support smaller/sampled reference datasets; allow user‑provided reference file.
- Use streaming or small subsets when pulling large datasets (e.g., PG19) and expose an input for the user to upload a custom reference list.

6) API and packaging cleanup
- Add lightweight, file‑free wrappers:
  - `preprocess(df_or_path, ...) -> df`
  - `detect_contamination_df(df, ref_texts_or_path, ...) -> df_with_flags`
  - `membership_inference_df(df, ...) -> df_with_flags_and_metrics`
  - `aggregate_flags_df(preprocessed_df, contam_df, member_df) -> flagged_df`
- Convert script‑local imports to package‑relative imports (`from .reference_comparison import ...`).
- Add `pyproject.toml` and an optional CLI (`typer`) to run subcommands.

7) Testing and quality
- Add small smoke tests that run on a tiny sample (no network) verifying:
  - cleaning/tokenization/segmentation shape properties
  - contamination detector fallbacks on tiny reference list
  - membership inference flags on synthetic duplicates/outliers
- Add `ruff` or `flake8` and `black` via pre‑commit.

8) Documentation
- Update README with accurate paths, Streamlit usage, and a “quickstart (no‑network)” mode using local toy data.
- Rename folders with spaces to underscored names to avoid tooling issues (optional).

---

## Streamlit App Plan

Goal: A guided, multi‑step app that preprocesses data, detects issues, previews flags, and produces a sanitized dataset and log for download.

Proposed pages/sections
- Data Input
  - Upload CSV (with a `text` or `segments` column) or select “use demo sample”.
  - Optional: upload reference file (one text per line) for contamination checks.
- Preprocess
  - Options: remove stopwords, segmentation mode (sentence/fixed), segment limit, simulate contamination (off by default).
  - Output preview: counts, sample rows.
- Contamination Detection
  - Options: reference similarity threshold, perplexity ratio threshold, model choices; “lightweight mode” toggles to avoid heavy downloads.
  - Output: summary metrics and flagged counts.
- Membership Inference
  - Options: embedding model, batch size, n_neighbors, thresholds.
  - Output: flagged duplicates/outliers + three plots (histogram, bar, scatter).
- Aggregate & Sanitize
  - Choose action (remove/anonymize/rewrite), preview diffs, export sanitized CSV and log CSV.

Key technical design
- Reuse module code in‑process to avoid shell calls and to surface real‑time progress.
- Cache heavy resources:
  - `@st.cache_resource` for models/tokenizers/embedders
  - `@st.cache_data` for DataFrame transformations with stable keys
- Use stable `segment_id` for merges across steps.
- Gracefully handle missing NLTK data by downloading once and caching.
- Offer “demo mode” with small bundled texts and a tiny reference list to avoid network on first run.

Minimal skeleton (illustrative)
```python
# streamlit_app.py (sketch)
import streamlit as st
import pandas as pd
from hashlib import sha256

st.set_page_config(page_title="Data Sanitization", layout="wide")

def make_id(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()[:16]

@st.cache_data
def preprocess_df(df: pd.DataFrame, remove_stopwords: bool, segment_mode: str):
    # call into src.preprocessor functions here
    # return df with a 'segments' column and a 'segment_id'
    df = df.copy()
    if "segments" not in df.columns and "text" in df.columns:
        df["segments"] = df["text"].astype(str)
    df["segment_id"] = df["segments"].astype(str).map(make_id)
    return df

st.title("Data Sanitization Pipeline")
data_file = st.file_uploader("Upload CSV with a text/segments column", type=["csv"]) 
if data_file:
    df = pd.read_csv(data_file)
    st.write("Input preview", df.head())
    df_prep = preprocess_df(df, remove_stopwords=False, segment_mode="sentence")
    st.write("Preprocessed preview", df_prep.head())
```

Deployment options
- Local: `streamlit run streamlit_app.py`
- Streamlit Community Cloud: push app + `requirements.txt` or conda `environment.yml` with dependencies; set `streamlit_app.py` as entrypoint.
- Container (optional later): Dockerfile with `conda` or `pip`, model cache mount, and GPU if available.

---

## Environment Setup (test_env)

Target: use the existing Conda environment `test_env`. Recommended adds (choose conda‑forge where possible; fall back to pip):

Conda (preferred)
```
conda activate test_env
conda install -y -c pytorch pytorch torchvision torchaudio  # CPU/MPS as appropriate
conda install -y -c conda-forge scikit-learn faiss-cpu sentence-transformers
```

Pip (if needed)
```
pip install streamlit sentence-transformers scikit-learn faiss-cpu
```

NLTK data (one‑time)
```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

Note: If MPS/GPU is present (Apple Silicon), ensure the correct `pytorch` build. For Linux CUDA, prefer `pytorch-cuda` from `nvidia` channel.

---

## Suggested Milestones
1. Hardening (Day 1–2)
   - Add missing deps; add `punkt` download
   - Introduce `segment_id`; unify data/paths; make contamination simulation opt‑in
   - Replace index joins with ID‑based merges
2. Streamlit MVP (Day 2–3)
   - Create app skeleton; in‑process wrappers; caching; “demo mode”
   - Basic results preview + downloads
3. Quality + DX (Day 3–4)
   - Smoke tests for each module; pre‑commit formatting/linting; better logs
   - README refresh; instructions for Streamlit Cloud

---

## Open Questions
- Which deployment target do you prefer: local, Streamlit Cloud, or containerized?
- What size datasets should the default “demo mode” ship with? Is fully offline a hard requirement?
- Do we need a stronger anonymization/rewriting policy (PII detection/redaction), or is “remove/anonymize placeholder/append tag” sufficient for now?

---

## Quick Commands Reference
- Full pipeline (current):
  - `python3 src/sanitization_main.py --full-pipeline --use-default-raw-data --sanitization-action anonymize`
- Streamlit (after adding `streamlit_app.py`):
  - `streamlit run streamlit_app.py`

---

## Summary
This plan builds on the project’s solid modular foundation. By shoring up dependencies, adding stable IDs, cleaning up paths/config, and providing thin function wrappers, we can power a responsive Streamlit experience without major refactors. The Streamlit MVP will let users upload data, tweak thresholds, inspect flags, and download sanitized results, while caching heavy resources to keep runtimes tractable.

