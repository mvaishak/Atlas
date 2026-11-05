"""A versioned analyzer shared by BM25 indexing, offline scoring, and HTTP search."""

import re

from atlasrank.data.splits import normalize_query

ANALYZER = "nfkc_casefold_unicode_alnum_v1"
TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def analyze(text: str) -> list[str]:
    """Retain Unicode letters/digits; punctuation and underscores separate terms.

    No stopwords, stemming, accent removal, or title/query truncation. Repeated
    query terms contribute repeatedly, matching the pinned BM25 library.
    """
    return TOKEN.findall(normalize_query(text))
