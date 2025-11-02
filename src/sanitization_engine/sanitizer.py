import logging
import hashlib
import pandas as pd
from tqdm import tqdm

def aggregate_flags(df_contamination, df_membership):
    """
    Aggregate flags from contamination and membership modules.

    Prefer segment_id-based aggregation when available; fall back to index.
    Returns a set of keys (segment_ids or indices) and a reason map keyed the same way.
    """
    logging.info("Aggregating contamination and membership inference flags.")

    use_ids = ('segment_id' in df_contamination.columns) and ('segment_id' in df_membership.columns)

    if use_ids:
        contamination_keys = set(
            df_contamination.loc[df_contamination['contamination_flag'], 'segment_id'].astype(str).tolist()
        )
        membership_keys = set(
            df_membership.loc[df_membership['membership_inference_flag'], 'segment_id'].astype(str).tolist()
        )
    else:
        contamination_keys = set(df_contamination[df_contamination['contamination_flag']].index.tolist())
        membership_keys = set(df_membership[df_membership['membership_inference_flag']].index.tolist())

    combined = contamination_keys.union(membership_keys)

    reason = {}
    for k in combined:
        reasons = []
        if k in contamination_keys:
            reasons.append("contamination")
        if k in membership_keys:
            reasons.append("membership")
        reason[k] = ", ".join(reasons)

    return combined, reason

def sanitize_data(df, flagged_keys, flag_reason, action="remove"):
    """
    Apply sanitization actions on rows identified by flagged_keys.

    flagged_keys can be dataframe indices (ints) or segment_ids (strs).
    """
    log_entries = []

    # Ensure segment_id exists for id-based operations
    if 'segment_id' not in df.columns:
        df['segment_id'] = df['segments'].astype(str).map(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()[:16])

    # Resolve keys to row indices
    target_indices = []
    for k in flagged_keys:
        if isinstance(k, (int, float)) and not isinstance(k, bool):
            try:
                idx = int(k)
                if 0 <= idx < len(df):
                    target_indices.append(idx)
            except Exception:
                continue
        else:  # treat as segment_id
            matches = df.index[df['segment_id'].astype(str) == str(k)].tolist()
            target_indices.extend(matches)

    # Deduplicate indices to avoid repeated operations
    target_indices = sorted(set(target_indices))

    for idx in tqdm(target_indices, desc="Sanitizing Data", dynamic_ncols=True):
        try:
            original_text = df.at[idx, 'segments']
        except Exception:
            continue
        # choose reason by original key if available; else unknown
        reason = flag_reason.get(idx) or flag_reason.get(df.at[idx, 'segment_id'], "unknown")

        if action == "remove":
            df.drop(idx, inplace=True)
            log_entries.append((idx, "removed", reason, original_text))
        elif action == "anonymize":
            df.at[idx, 'segments'] = "[REMOVED DUE TO PRIVACY RISK]"
            log_entries.append((idx, "anonymized", reason, original_text))
        elif action == "rewrite":
            df.at[idx, 'segments'] = original_text + " [REWRITTEN]"
            log_entries.append((idx, "rewritten", reason, original_text))
        else:
            logging.warning(f"Invalid action: {action}, skipping index {idx}")

    log_df = pd.DataFrame(log_entries, columns=["index", "action", "reason", "original_text"])
    return df.reset_index(drop=True), log_df
