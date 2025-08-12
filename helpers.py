import re
import pandas as pd
import numpy as np
"""
clean_instagram_utils.py

Usage:
- Edit `NUMERIC_COL_CANDIDATES` if you want to force certain names to be treated as numeric.
- Call `scan_and_clean(df, numeric_candidates=..., save_clean_path="cleaned.csv")`
  which returns (clean_df, report_df).
"""


# ---- Helpers / regexes ----
EMOJI_RE = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

# Keep # and @ when cleaning caption (we'll extract them separately)
SPECIAL_KEEP_RE = re.compile(r'[^0-9A-Za-z\s#@_]')  # characters to remove in "caption_clean_keep_hashtags"

# For cleaning numeric-looking strings (remove commas, spaces, parentheses, currency signs)
NUMERIC_CLEAN_RE_COMMAS = re.compile(r'[,\s]+')
NUMERIC_CLEAN_RE_NON_NUM = re.compile(r'[^\d\.\-]')  # keep digits, dot, minus

# ---- Normalization ----
def normalize_columns(df):
    """Lowercase, strip, replace spaces with underscore, drop odd chars from column names."""
    cols = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    df = df.copy()
    df.columns = cols
    return df

# ---- Numeric cleaning/coercion ----
def coerce_numeric_series(s: pd.Series):
    """
    Attempt to clean a Series and convert it to numeric.
    Returns (numeric_series, non_numeric_mask)
    - numeric_series: the converted pd.Series (dtype float)
    - non_numeric_mask: boolean Series indicating original non-numeric values that couldn't be converted
    """
    s_orig = s.copy()
    # Mark empty-like as NaN
    s_str = s_orig.astype(str).replace({'nan': np.nan, 'None': np.nan, 'none': np.nan})
    # Detect percent signs (we'll divide by 100 later)
    had_percent = s_str.str.contains('%', na=False)
    # Remove commas and whitespace
    temp = s_str.fillna("").astype(str)
    temp = NUMERIC_CLEAN_RE_COMMAS.sub("", temp)            # remove commas and whitespace
    # Parentheses like (123) -> -123
    temp = temp.replace(to_replace=r'^\((.*)\)$', value=r'-\1', regex=True)
    # Remove currency or other stray characters, keep digits/dot/minus
    temp = NUMERIC_CLEAN_RE_NON_NUM.sub("", temp)
    # Convert to numeric
    numeric = pd.to_numeric(temp, errors='coerce')
    # adjust percent values
    numeric.loc[had_percent & numeric.notna()] = numeric.loc[had_percent & numeric.notna()] / 100.0
    # Determine which rows were non-numeric originally but non-empty
    non_numeric_mask = s_orig.notna() & numeric.isna()
    return numeric, non_numeric_mask

# ---- Text scans ----
def count_emojis(text):
    if pd.isna(text):
        return 0
    return len(EMOJI_RE.findall(str(text)))

def non_ascii_ratio(text):
    if pd.isna(text) or len(str(text)) == 0:
        return 0.0
    chars = [ch for ch in str(text)]
    non_ascii = sum(1 for ch in chars if ord(ch) > 127)
    return non_ascii / len(chars)

def special_char_fraction(text):
    """Return fraction of characters that are not alnum, whitespace, # or @ (useful to detect noisy cells)"""
    if pd.isna(text) or len(str(text)) == 0:
        return 0.0
    total = len(str(text))
    special = sum(1 for ch in str(text) if not (ch.isalnum() or ch.isspace() or ch in ['#','@','_']))
    return special / total

# ---- Hashtags and mentions extraction ----
def extract_hashtags(text):
    if pd.isna(text):
        return []
    return re.findall(r'#[\w_]+', str(text).lower())

def extract_mentions(text):
    if pd.isna(text):
        return []
    return re.findall(r'@[\w_]+', str(text).lower())

# ---- Caption cleaning choices ----
def clean_caption_keep_hashtags_mentions(text):
    """Remove emojis and other special symbols but keep # and @ and underscores."""
    if pd.isna(text):
        return ""
    t = EMOJI_RE.sub("", str(text))           # remove emojis
    t = SPECIAL_KEEP_RE.sub("", t)            # remove other special chars but keep #,@,_
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def clean_caption_strict(text):
    """Remove emojis and all punctuation (letters, numbers, spaces only)."""
    if pd.isna(text):
        return ""
    t = EMOJI_RE.sub("", str(text))
    t = re.sub(r'[^\w\s]', '', t)   # remove punctuation
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ---- Main pipeline function ----
def scan_and_clean(
    df,
    numeric_candidates=None,
    save_clean_path=None,
    save_report_path=None,
    drop_rows_with_unfixable_numeric=False,
    impute_numeric_with='median'   # or 'mean' or None (leave NaNs)
):
    """
    Scan all columns for unwanted characters and try to clean numeric columns automatically.
    Returns (clean_df, report_df)
    - numeric_candidates: list of column names you want treated as numeric (after normalization)
      If None, the function will try to infer numeric-like columns by name (keywords).
    - drop_rows_with_unfixable_numeric: if True, drop rows that failed numeric coercion.
    - impute_numeric_with: 'median', 'mean', or None
    """
    df = normalize_columns(df)
    out = df.copy()
    report = []

    # heuristics: if numeric_candidates not provided, infer by common keywords
    if numeric_candidates is None:
        keywords = ['impress', 'views', 'from_home', 'from_hashtag', 'from_hashtags', 'from_explore',
                    'likes', 'comments', 'saves', 'shares', 'profile', 'follow']
        numeric_candidates = [c for c in out.columns if any(k in c for k in keywords)]

    # 1) Numeric cleaning & validation
    numeric_report_rows = []
    for col in numeric_candidates:
        if col not in out.columns:
            continue
        numeric_series, non_numeric_mask = coerce_numeric_series(out[col])
        n_bad = non_numeric_mask.sum()
        pct_bad = n_bad / len(out) if len(out) > 0 else 0
        # attach cleaned numeric column (keep as float); keep raw as backup
        out[col + "_raw"] = out[col]
        out[col] = numeric_series

        # optionally impute
        if impute_numeric_with in ('median','mean'):
            if impute_numeric_with == 'median':
                fill = out[col].median(skipna=True)
            else:
                fill = out[col].mean(skipna=True)
            out[col] = out[col].fillna(fill)

        numeric_report_rows.append({
            "column": col,
            "n_bad": int(n_bad),
            "pct_bad": float(pct_bad),
            "action": "coerced; imputed="+str(impute_numeric_with) if impute_numeric_with else "coerced; left NaN"
        })

        # optionally drop rows where non_numeric_mask True
        if drop_rows_with_unfixable_numeric and n_bad > 0:
            out = out.loc[~non_numeric_mask].reset_index(drop=True)

    # 2) Text scanning for every object/string column
    text_report_rows = []
    text_cols = [c for c in out.columns if out[c].dtype == object or pd.api.types.is_string_dtype(out[c])]
    for col in text_cols:
        sample_nonempty = out[col].dropna().astype(str)
        n_nonempty = len(sample_nonempty)
        emoji_counts = sample_nonempty.apply(count_emojis)
        total_emoji = int(emoji_counts.sum())
        avg_emoji_per_cell = float(emoji_counts.mean()) if n_nonempty>0 else 0.0
        non_ascii_ratios = sample_nonempty.apply(non_ascii_ratio)
        avg_non_ascii = float(non_ascii_ratios.mean()) if n_nonempty>0 else 0.0
        special_frac = sample_nonempty.apply(special_char_fraction)
        avg_special_frac = float(special_frac.mean()) if n_nonempty>0 else 0.0

        text_report_rows.append({
            "column": col,
            "n_nonempty": int(n_nonempty),
            "total_emoji": total_emoji,
            "avg_emoji_per_cell": avg_emoji_per_cell,
            "avg_non_ascii_ratio": avg_non_ascii,
            "avg_special_char_fraction": avg_special_frac
        })

    # 3) Create cleaned caption/hashtags fields if present
    if 'caption' in out.columns:
        out['caption_raw'] = out['caption'].astype(str)
        out['caption_noemoji_keep_hashtags'] = out['caption'].apply(clean_caption_keep_hashtags_mentions)
        out['caption_strict'] = out['caption'].apply(clean_caption_strict)
        out['caption_emoji_count'] = out['caption'].apply(count_emojis)
        out['caption_hashtags_extracted'] = out['caption'].apply(extract_hashtags)
        out['caption_mentions_extracted'] = out['caption'].apply(extract_mentions)
        out['caption_hashtag_count'] = out['caption_hashtags_extracted'].apply(len)

    # If there is a 'hashtags' column, normalize it (also keep raw)
    if 'hashtags' in out.columns:
        out['hashtags_raw'] = out['hashtags'].astype(str)
        # try to extract tokens from this column (commas, spaces, semicolons)
        def parse_hashtags_field(x):
            if pd.isna(x): return []
            s = str(x).lower()
            # find explicit #tags
            tags = re.findall(r'#[\w_]+', s)
            if tags:
                return tags
            # otherwise split by comma/space/semicolon
            parts = re.split(r'[,\n;]+', s)
            parts = [p.strip().lower() for p in parts if p.strip()!='']
            # ensure tokens start with # for consistency
            parts = [('#' + p) if not p.startswith('#') else p for p in parts]
            return parts
        out['hashtags_extracted'] = out['hashtags_raw'].apply(parse_hashtags_field)
        out['hashtags_count'] = out['hashtags_extracted'].apply(len)

    # 4) Normalize content_type column
    if 'content_type' in out.columns:
        out['content_type_raw'] = out['content_type'].astype(str)
        out['content_type'] = out['content_type'].str.lower().str.strip()
        # map variants to main categories (extend if needed)
        mapping = {
            'reel': 'reel', 'r': 'reel',
            'video': 'video', 'vid': 'video',
            'image': 'image', 'photo': 'image', 'img': 'image',
            'carousel': 'carousel'
        }
        out['content_type'] = out['content_type'].replace(mapping)
        # one-hot encode
        dummies = pd.get_dummies(out['content_type'], prefix='type', dummy_na=False)
        out = pd.concat([out, dummies], axis=1)

    # 5) Prepare report DataFrame
    report_rows = []
    report_rows.extend(numeric_report_rows)
    # extend with text report rows but rename keys to be consistent
    for r in text_report_rows:
        report_rows.append({
            "column": r['column'],
            "n_bad": None,
            "pct_bad": None,
            "action": None,
            "n_nonempty": r['n_nonempty'],
            "total_emoji": r['total_emoji'],
            "avg_emoji_per_cell": r['avg_emoji_per_cell'],
            "avg_non_ascii_ratio": r['avg_non_ascii_ratio'],
            "avg_special_char_fraction": r['avg_special_char_fraction']
        })

    report_df = pd.DataFrame(report_rows).fillna(value=np.nan)

    # Save if requested
    if save_clean_path:
        out.to_csv(save_clean_path, index=False)
    if save_report_path:
        report_df.to_csv(save_report_path, index=False)

    return out, report_df
