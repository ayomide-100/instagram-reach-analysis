import re
import pandas as pd

def count_hashtags(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"#\w+", text))

def caption_length_chars(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.strip())

def compute_hashtag_density(hashtag_count: int, caption_len: int) -> float:
    # density as count per character; avoids div-by-zero
    return float(hashtag_count) / max(1, caption_len)
