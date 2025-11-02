import io
import os
from hashlib import sha256
from typing import List, Optional

# Standard imports; environment is configured to provide pyarrow
import pandas as pd
import streamlit as st

# Local imports from the project
from src.preprocessor.cleaning import normalize_text
from src.preprocessor.segmentation import segment_dataframe
from src.membership_inference_checker.neighborhood import flag_membership
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.sanitization_engine.sanitizer import aggregate_flags, sanitize_data


st.set_page_config(page_title="Data Sanitization Pipeline", layout="wide")


def _make_segment_id(s: str) -> str:
    return sha256(str(s).encode("utf-8")).hexdigest()[:16]


@st.cache_data(show_spinner=False)
def preprocess_df(
    df_in: pd.DataFrame,
    remove_stopwords: bool = False,
    segment_mode: str = "sentence",
) -> pd.DataFrame:
    df = df_in.copy()
    # Pick input column
    if "segments" in df.columns:
        base_col = "segments"
        df[base_col] = df[base_col].astype(str)
        cleaned = df[base_col].map(lambda x: normalize_text(x, remove_stopwords=remove_stopwords))
        df["segments"] = cleaned
    elif "text" in df.columns:
        df["text"] = df["text"].astype(str)
        df["cleaned_text"] = df["text"].map(lambda x: normalize_text(x, remove_stopwords=remove_stopwords))
        df = segment_dataframe(df, text_column="cleaned_text", mode=segment_mode)
        df.drop(columns=[c for c in ["text", "cleaned_text"] if c in df.columns], inplace=True, errors="ignore")
    else:
        # Fallback: assume whole rows are text-like
        df["segments"] = df.apply(lambda r: normalize_text(" ".join(map(str, r.values)), remove_stopwords=remove_stopwords), axis=1)

    df = df[["segments"]].copy()
    df["segment_id"] = df["segments"].map(_make_segment_id)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(show_spinner=False)
def detect_contamination_lite(
    df: pd.DataFrame,
    reference_texts: List[str],
    sim_threshold: float = 0.9,
) -> pd.DataFrame:
    # TF-IDF based similarity as a lightweight, offline-friendly fallback
    texts = [str(t) for t in df["segments"].tolist()]
    refs = [str(t) for t in reference_texts]
    if not refs:
        refs = ["reference text"]
    vect = TfidfVectorizer(max_features=20000)
    ref_mat = vect.fit_transform(refs)
    seg_mat = vect.transform(texts)
    sim = cosine_similarity(seg_mat, ref_mat)
    max_sim = sim.max(axis=1) if sim.size else np.zeros(len(texts))
    out = df.copy()
    out["ref_similarity"] = max_sim
    out["ref_flag"] = (max_sim >= sim_threshold)
    out["confidence_flag"] = False
    out["contamination_flag"] = out["ref_flag"]
    return out


@st.cache_data(show_spinner=False)
def membership_inference_df(
    df: pd.DataFrame,
    embedding_model: str = "tfidf",  # default to tf-idf for portability
    batch_size: int = 32,
    n_neighbors: int = 6,
    high_sim_threshold: float = 0.95,
    low_sim_threshold: float = 0.3,
) -> pd.DataFrame:
    # Try sentence-transformers; fallback to TF-IDF embeddings
    texts = df["segments"].astype(str).tolist()
    emb = None
    if embedding_model.lower() != "tfidf":
        try:  # defer heavy import and allow failure gracefully
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(embedding_model)
            emb = np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True))
        except Exception:
            emb = None
    if emb is None:
        vect = TfidfVectorizer(max_features=50000)
        emb = vect.fit_transform(texts).astype("float32").toarray()
    dup_flags, out_flags, max_sim = flag_membership(emb, high_sim_threshold=high_sim_threshold, low_sim_threshold=low_sim_threshold, n_neighbors=n_neighbors)
    out = df.copy()
    out["max_neighbor_similarity"] = max_sim
    out["duplicate_flag"] = dup_flags
    out["outlier_flag"] = out_flags
    out["membership_inference_flag"] = out["duplicate_flag"] | out["outlier_flag"]
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


st.title("Data Sanitization Pipeline")
st.caption("Preprocess → Detect Contamination → Membership Inference → Sanitize")

with st.sidebar:
    step = st.radio(
        "Step",
        ["1. Data", "2. Preprocess", "3. Contamination", "4. Membership", "5. Sanitize"],
        index=0,
    )
    st.markdown("---")
    st.markdown("Tip: Use demo data to try the app quickly.")


# Session state containers
for key in ["df_raw", "df_prep", "df_contam", "df_member", "df_sanitized", "df_sanitize_log"]:
    if key not in st.session_state:
        st.session_state[key] = None


def show_data_step():
    st.subheader("Upload or Use Demo Data")
    uploaded = st.file_uploader("Upload CSV with a 'text' or 'segments' column", type=["csv"]) 

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state.df_raw = df
        st.success(f"Loaded {len(df)} rows from upload")
        show_df(df, rows=10)
    else:
        if st.button("Use demo data"):
            demo_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Deep learning models often require large datasets.",
                "A reference benchmark should not leak into training data.",
                "Sanitization removes or anonymizes risky content.",
                "Machine learning evaluates performance on withheld test sets.",
                "Data duplication can bias model outcomes.",
                "Neighbors in embedding space indicate similarity.",
                "Streamlit makes building data apps straightforward.",
            ]
            st.session_state.df_raw = pd.DataFrame({"text": demo_texts})
            st.success("Demo data loaded (8 rows)")
            show_df(st.session_state.df_raw, rows=10)


def show_preprocess_step():
    st.subheader("Preprocess")
    if st.session_state.df_raw is None:
        st.info("Please load data in Step 1")
        return

    c1, c2 = st.columns(2)
    with c1:
        remove_stop = st.checkbox("Remove stopwords", value=False)
        seg_mode = st.selectbox("Segmentation mode", ["sentence", "fixed", "none"], index=0)
    with c2:
        st.caption("Segmentation is applied only when starting from a 'text' column")

    if st.button("Run preprocessing"):
        with st.spinner("Preprocessing..."):
            df_prep = preprocess_df(st.session_state.df_raw, remove_stopwords=remove_stop, segment_mode=seg_mode)
        st.session_state.df_prep = df_prep
        st.success(f"Preprocessed segments: {len(df_prep)}")
        show_df(df_prep, rows=20)

    if st.session_state.df_prep is not None:
        show_df(st.session_state.df_prep, rows=20)


def show_contamination_step():
    st.subheader("Contamination Detection (Lite)")
    if st.session_state.df_prep is None:
        st.info("Please preprocess data in Step 2")
        return

    st.caption("Lite mode checks similarity against a small user-provided reference list.")
    ref_texts_input = st.text_area(
        "Reference texts (one per line)",
        value="""This is a known contaminated text.
Benchmark dataset entry should not appear in training.
Another clean sample line for reference.""",
        height=120,
    )
    ref_texts = [l.strip() for l in ref_texts_input.splitlines() if l.strip()]
    sim_thr = st.slider("Reference similarity threshold", 0.5, 0.99, 0.9, 0.01)
    ref_model = st.text_input("Reference embedding model", value="all-MiniLM-L6-v2")

    if st.button("Run contamination detection"):
        try:
            with st.spinner("Computing reference similarity..."):
                df_contam = detect_contamination_lite(st.session_state.df_prep, ref_texts, ref_model_name=ref_model, sim_threshold=sim_thr)
            st.session_state.df_contam = df_contam
            flagged = int(df_contam["contamination_flag"].sum())
            st.success(f"Contamination flags: {flagged} / {len(df_contam)}")
            show_df(df_contam, rows=20)
        except Exception as e:
            st.error(f"Contamination step failed: {e}")

    if st.session_state.df_contam is not None:
        show_df(st.session_state.df_contam, rows=20)


def show_membership_step():
    st.subheader("Membership Inference")
    if st.session_state.df_prep is None:
        st.info("Please preprocess data in Step 2")
        return

    emb_model = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
    n_neighbors = st.slider("Nearest neighbors", 3, 20, 6)
    high_thr = st.slider("High-sim threshold (duplicates)", 0.5, 0.999, 0.95, 0.01)
    low_thr = st.slider("Low-sim threshold (outliers)", 0.0, 0.7, 0.3, 0.01)
    batch = st.slider("Batch size", 8, 128, 32, 8)

    if st.button("Run membership analysis"):
        try:
            with st.spinner("Embedding and neighborhood analysis..."):
                df_member = membership_inference_df(
                    st.session_state.df_prep,
                    embedding_model=emb_model,
                    batch_size=batch,
                    n_neighbors=n_neighbors,
                    high_sim_threshold=high_thr,
                    low_sim_threshold=low_thr,
                )
            st.session_state.df_member = df_member
            flagged = int(df_member["membership_inference_flag"].sum())
            st.success(f"Membership flags: {flagged} / {len(df_member)}")
            show_df(df_member, rows=20)
        except Exception as e:
            st.error(f"Membership step failed: {e}")

    if st.session_state.df_member is not None:
        show_df(st.session_state.df_member, rows=20)


def show_sanitize_step():
    st.subheader("Aggregate & Sanitize")
    if st.session_state.df_prep is None:
        st.info("Please preprocess data in Step 2")
        return
    if st.session_state.df_contam is None and st.session_state.df_member is None:
        st.info("Run contamination and/or membership steps to generate flags")
        return

    action = st.selectbox("Sanitization action", ["remove", "anonymize", "rewrite"], index=0)

    if st.button("Run sanitization"):
        try:
            df_contam = st.session_state.df_contam if st.session_state.df_contam is not None else st.session_state.df_prep.assign(contamination_flag=False)
            df_member = st.session_state.df_member if st.session_state.df_member is not None else st.session_state.df_prep.assign(membership_inference_flag=False)

            flagged_indices, flag_reason = aggregate_flags(df_contam, df_member)
            st.info(f"Flagged segments: {len(flagged_indices)}")

            sanitized_df, log_df = sanitize_data(st.session_state.df_prep.copy(), flagged_indices, flag_reason, action)
            st.session_state.df_sanitized = sanitized_df
            st.session_state.df_sanitize_log = log_df

            st.success("Sanitization complete")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Sanitized sample")
                show_df(sanitized_df, rows=20)
            with c2:
                st.caption("Sanitization log sample")
                show_df(log_df, rows=20)

            st.download_button("Download sanitized CSV", data=to_csv_bytes(sanitized_df), file_name="sanitized_dataset.csv", mime="text/csv")
            st.download_button("Download log CSV", data=to_csv_bytes(log_df), file_name="sanitization_log.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Sanitization failed: {e}")

    if st.session_state.df_sanitized is not None:
        show_df(st.session_state.df_sanitized, rows=20)


if step.startswith("1"):
    show_data_step()
elif step.startswith("2"):
    show_preprocess_step()
elif step.startswith("3"):
    show_contamination_step()
elif step.startswith("4"):
    show_membership_step()
else:
    show_sanitize_step()
def show_df(df: pd.DataFrame, rows: int = 20, key: Optional[str] = None):
    try:
        st.dataframe(df.head(rows), use_container_width=True)
    except Exception as e:  # pyarrow may be broken; provide a safe fallback
        st.warning(f"Fallback table rendering due to display backend issue: {e}")
        st.text(df.head(rows).to_string())
