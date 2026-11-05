"""Shared fixed-length text/token contract and a deterministic token-ID audit hash."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from atlasrank.data.splits import normalize_query


def tokenize_fixed(tokenizer: Any, texts: Sequence[str]) -> Any:
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("Tokenization inputs must be a nonempty sequence of strings")
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer(
        [normalize_query(text) for text in texts],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )


def token_id_fingerprint(tokenizer: Any, texts: Sequence[str]) -> dict[str, Any]:
    unique = sorted({normalize_query(text) for text in texts})
    if not unique:
        raise ValueError("Token-ID fingerprint requires at least one text")
    digest = hashlib.sha256()
    for offset in range(0, len(unique), 128):
        batch = unique[offset : offset + 128]
        encoded = tokenize_fixed(tokenizer, batch)
        rows = {
            name: tensor.tolist()
            for name, tensor in sorted(encoded.items())
            if hasattr(tensor, "tolist")
        }
        for row_index, text in enumerate(batch):
            token_fields = {name: values[row_index] for name, values in rows.items()}
            line = json.dumps(
                [text, token_fields],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            digest.update(line.encode("utf-8") + b"\n")
    return {"sha256": digest.hexdigest(), "unique_texts": len(unique), "max_length": 256}
