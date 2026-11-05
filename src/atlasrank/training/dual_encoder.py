"""Deterministic train-only dual-encoder sampling and contrastive objective."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class GroupJudgments:
    locale: str
    group_key: str
    group_sha256: str
    query: str
    positives: tuple[str, ...]
    negatives: tuple[str, ...]


@dataclass(frozen=True)
class PlannedGroup:
    locale: str
    group_key: str
    group_sha256: str
    query: str
    positives: tuple[str, ...]
    negatives: tuple[str, ...]


@dataclass(frozen=True)
class SamplingPlan:
    schema_version: int
    seed: int
    max_positives: int
    max_negatives: int
    source_group_count: int
    usable_group_count: int
    excluded_group_count: int
    globally_masked_negative_count: int
    groups: tuple[PlannedGroup, ...]

    def canonical_bytes(self) -> bytes:
        value = asdict(self)
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _order(
    seed: int, group: GroupJudgments, role: str, products: tuple[str, ...]
) -> tuple[str, ...]:
    prefix = f"atlasrank-m5-v1\0{seed}\0{group.locale}\0{group.group_key}\0{role}\0"
    return tuple(
        sorted(
            set(products),
            key=lambda product: (
                hashlib.sha256((prefix + product).encode()).digest(),
                product,
            ),
        )
    )


def build_sampling_plan(
    groups: list[GroupJudgments], *, seed: int = 42, max_positives: int = 2, max_negatives: int = 4
) -> SamplingPlan:
    """Freeze one deterministic group/pair order and enforce train-global E masking."""
    if seed < 0 or max_positives < 1 or max_negatives < 1:
        raise ValueError("Invalid sampling-plan seed or limits")
    keys = [(group.locale, group.group_key) for group in groups]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate training query group")
    if any(not g.locale or not g.group_key or not g.group_sha256 or not g.query for g in groups):
        raise ValueError("Training group is missing canonical identity or query text")
    if any(not p for g in groups for p in (*g.positives, *g.negatives)):
        raise ValueError("Training judgments contain an empty product identity")

    global_positives = {p for group in groups for p in group.positives}
    planned: list[PlannedGroup] = []
    masked_count = 0
    for group in sorted(
        groups,
        key=lambda g: (
            hashlib.sha256(
                f"atlasrank-m5-v1\0{seed}\0{g.locale}\0{g.group_key}\0group".encode()
            ).digest()
            + f"\0{g.locale}\0{g.group_key}".encode()
        ),
    ):
        valid_negatives = tuple(p for p in group.negatives if p not in global_positives)
        masked_count += len(set(group.negatives).difference(valid_negatives))
        positives = _order(seed, group, "positive", group.positives)[:max_positives]
        negatives = _order(seed, group, "negative", valid_negatives)[:max_negatives]
        if not positives or not negatives:
            continue
        planned.append(
            PlannedGroup(
                group.locale,
                group.group_key,
                group.group_sha256,
                group.query,
                positives,
                negatives,
            )
        )
    return SamplingPlan(
        schema_version=1,
        seed=seed,
        max_positives=max_positives,
        max_negatives=max_negatives,
        source_group_count=len(groups),
        usable_group_count=len(planned),
        excluded_group_count=len(groups) - len(planned),
        globally_masked_negative_count=masked_count,
        groups=tuple(planned),
    )


def contrastive_group_loss(logits: Tensor, positive_mask: Tensor, valid_mask: Tensor) -> Tensor:
    """Equal-mean query loss with a summed-positive log-sum-exp numerator.

    Per group: `logsumexp(all valid logits) - logsumexp(all positive logits)`.
    Positive logits therefore form one summed numerator event and are not treated
    as negatives for one another. Every row has equal weight in the batch mean.
    """
    if logits.ndim != 2 or positive_mask.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ValueError("Logits and masks must be equally shaped rank-2 tensors")
    if positive_mask.dtype != torch.bool or valid_mask.dtype != torch.bool:
        raise ValueError("Contrastive masks must be boolean")
    if bool((positive_mask & ~valid_mask).any()):
        raise ValueError("Positive candidates must be valid")
    if bool((~positive_mask.any(dim=1)).any()):
        raise ValueError("Every group requires at least one positive")
    negative_counts = (valid_mask & ~positive_mask).sum(dim=1)
    if bool((negative_counts == 0).any()):
        raise ValueError("Every group requires at least one admissible negative")
    if not bool(torch.isfinite(logits[valid_mask]).all()):
        raise ValueError("Valid logits must be finite")
    denominator = torch.logsumexp(logits.masked_fill(~valid_mask, -torch.inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
    return (denominator - numerator).mean()


def cosine_logits(queries: Tensor, products: Tensor, temperature: float) -> Tensor:
    if temperature <= 0 or queries.ndim != 2 or products.ndim != 3:
        raise ValueError(
            "Expected positive temperature and [groups, dim]/[groups, candidates, dim]"
        )
    if queries.shape[0] != products.shape[0] or queries.shape[1] != products.shape[2]:
        raise ValueError("Query/product embedding shapes do not align")
    query_vectors = torch.nn.functional.normalize(queries.float(), p=2, dim=-1)
    product_vectors = torch.nn.functional.normalize(products.float(), p=2, dim=-1)
    return torch.einsum("bd,bnd->bn", query_vectors, product_vectors) / temperature


def batch_masks(
    positive_counts: list[int], negative_counts: list[int], *, device: Any
) -> tuple[Tensor, Tensor]:
    if len(positive_counts) != len(negative_counts) or not positive_counts:
        raise ValueError("Batch must contain matching nonempty group counts")
    if any(p < 1 or n < 1 for p, n in zip(positive_counts, negative_counts, strict=True)):
        raise ValueError("Every group needs at least one positive and one negative")
    width = max(p + n for p, n in zip(positive_counts, negative_counts, strict=True))
    positive = torch.zeros((len(positive_counts), width), dtype=torch.bool, device=device)
    valid = torch.zeros_like(positive)
    for row, (pos_count, neg_count) in enumerate(
        zip(positive_counts, negative_counts, strict=True)
    ):
        positive[row, :pos_count] = True
        valid[row, : pos_count + neg_count] = True
    return positive, valid
