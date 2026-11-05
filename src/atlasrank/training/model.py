"""Differentiable shared-weight MiniLM execution using the M4 text contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from atlasrank.training.dual_encoder import batch_masks, contrastive_group_loss, cosine_logits
from atlasrank.training.tokenization import tokenize_fixed


@dataclass(frozen=True)
class TrainingGroup:
    group_key: str
    query: str
    positive_titles: tuple[str, ...]
    negative_titles: tuple[str, ...]


class SharedMiniLM:
    """Wrap the pinned SentenceTransformer without changing its tokenizer/pooling."""

    def __init__(self, model_path: str, device: str = "cuda:0") -> None:
        from sentence_transformers import SentenceTransformer

        self.device = torch.device(device)
        self.model: Any = SentenceTransformer(
            model_path,
            device=device,
            local_files_only=True,
            trust_remote_code=False,
            model_kwargs={
                "use_safetensors": True,
                "dtype": torch.float32,
                "attn_implementation": "eager",
            },
        )
        if self.model.max_seq_length != 256 or self.model.get_sentence_embedding_dimension() != 384:
            raise ValueError("Pinned M4 MiniLM architecture/truncation does not match")
        self.model.tokenizer.padding_side = "right"
        self.model.tokenizer.truncation_side = "right"
        self.model.to(self.device)

    def _forward(self, texts: list[str]) -> Tensor:
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("Encoder inputs must be nonempty text strings")
        features = tokenize_fixed(self.model.tokenizer, texts)
        features = {name: value.to(self.device) for name, value in features.items()}
        output = cast(Tensor, self.model(features)["sentence_embedding"].float())
        return output

    def loss(self, groups: list[TrainingGroup], temperature: float = 0.05) -> Tensor:
        if not groups:
            raise ValueError("Cannot train on an empty batch")
        if any(not g.positive_titles or not g.negative_titles for g in groups):
            raise ValueError("Every training group requires positive and negative titles")
        queries = self._forward([group.query for group in groups])
        per_group_titles = [group.positive_titles + group.negative_titles for group in groups]
        counts = [len(values) for values in per_group_titles]
        products_flat = [title for values in per_group_titles for title in values]
        product_embeddings = self._forward(products_flat)
        width = max(counts)
        product_matrix = product_embeddings.new_zeros(
            (len(groups), width, product_embeddings.shape[1])
        )
        offset = 0
        pos_counts: list[int] = []
        neg_counts: list[int] = []
        for row, group in enumerate(groups):
            pos_count = len(group.positive_titles)
            count = counts[row]
            product_matrix[row, :count] = product_embeddings[offset : offset + count]
            offset += count
            pos_counts.append(pos_count)
            neg_counts.append(len(group.negative_titles))
        logits = cosine_logits(queries, product_matrix, temperature)
        positive_mask, valid_mask = batch_masks(pos_counts, neg_counts, device=self.device)
        return contrastive_group_loss(logits, positive_mask, valid_mask)
