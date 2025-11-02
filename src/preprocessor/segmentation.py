from nltk.tokenize import sent_tokenize


def segment_text(text, mode='sentence', fixed_token_length=100):
    """
    Segment text into units based on mode.

    Args:
        text (str): Input text.
        mode (str): 'sentence' for sentence segmentation,
                    'fixed' for fixed token-length segments,
                    'none' to return the text as a single segment.
        fixed_token_length (int): Token count per segment (for 'fixed' mode).

    Returns:
        list: List of text segments.
    """
    if mode == 'sentence':
        return sent_tokenize(text)
    elif mode == 'fixed':
        # Lazy import to avoid heavy transformer dependencies unless needed
        try:
            from .tokenization import tokenize_text as _tokenize_text
        except ImportError:  # pragma: no cover
            from tokenization import tokenize_text as _tokenize_text
        tokens = _tokenize_text(text)
        return [' '.join(tokens[i:i + fixed_token_length]) for i in range(0, len(tokens), fixed_token_length)]
    else:
        return [text]


def segment_dataframe(df, text_column='cleaned_text', mode='sentence', limit_segments=None):
    """
    Apply segmentation to each entry in a DataFrame column and explode the result.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Column containing the text to segment.
        mode (str): Segmentation mode.
        limit_segments (int|None): If provided, stop after producing this many segments.

    Returns:
        pd.DataFrame: DataFrame with a new column 'segments', exploded into one row per segment.
    """
    if limit_segments is None:
        df['segments'] = df[text_column].apply(lambda x: segment_text(x, mode=mode))
        return df.explode('segments')
    # Early limiting path: build rows until we reach the limit
    import pandas as _pd
    out_rows = []
    produced = 0
    for _, row in df.iterrows():
        segs = segment_text(row[text_column], mode=mode)
        for s in segs:
            new_row = row.to_dict()
            new_row['segments'] = s
            out_rows.append(new_row)
            produced += 1
            if produced >= limit_segments:
                return _pd.DataFrame(out_rows)
    return _pd.DataFrame(out_rows)
