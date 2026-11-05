from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from atlasrank.training.dual_encoder import (
    GroupJudgments,
    build_sampling_plan,
    contrastive_group_loss,
)
from atlasrank.training.model import SharedMiniLM, TrainingGroup
from atlasrank.training.tokenization import token_id_fingerprint, tokenize_fixed


def test_sampling_plan_is_stable_and_masks_global_train_positives() -> None:
    groups = [
        GroupJudgments("en_US", "q1", "q1", "phone", ("p1", "p2", "p3"), ("n1", "n2")),
        GroupJudgments("en_US", "q2", "q2", "case", ("p3",), ("n3", "p1")),
    ]
    plan = build_sampling_plan(groups, seed=42, max_positives=2, max_negatives=4)
    assert plan == build_sampling_plan(groups, seed=42, max_positives=2, max_negatives=4)
    q1 = next(group for group in plan.groups if group.group_key == "q1")
    assert len(q1.positives) == 2
    assert set(q1.positives).issubset({"p1", "p2", "p3"})
    assert "p1" not in q1.negatives and "p3" not in q1.negatives
    assert plan.globally_masked_negative_count == 1
    assert plan.usable_group_count == 2
    assert plan.sha256 == "d356bc54a12e82955fd8661f052d061cf377f85c55590b036a2a1995ff02d91c"
    assert q1.positives == ("p2", "p3")
    assert q1.negatives == ("n1", "n2")
    assert len(plan.sha256) == 64


def test_summed_positive_loss_matches_hand_calculation_with_multiple_positives() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    positive_mask = torch.tensor([[True, True, False]])
    valid_mask = torch.tensor([[True, True, True]])
    loss = contrastive_group_loss(logits, positive_mask, valid_mask)
    expected = -(
        math.log(math.exp(2.0) + math.exp(1.0)) - math.log(math.exp(2.0) + math.exp(1.0) + 1.0)
    )
    assert loss.item() == pytest.approx(expected, abs=1e-12)
    loss.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert logits.grad[0, 0].item() < 0
    assert logits.grad[0, 1].item() < 0
    assert logits.grad[0, 2].item() > 0


def test_invalid_candidates_have_zero_loss_contribution_and_gradient() -> None:
    logits = torch.tensor([[1.0, 0.0, 100.0]], dtype=torch.float64, requires_grad=True)
    positive_mask = torch.tensor([[True, False, False]])
    valid_mask = torch.tensor([[True, True, False]])
    loss = contrastive_group_loss(logits, positive_mask, valid_mask)
    loss.backward()  # type: ignore[no-untyped-call]
    assert loss.item() == pytest.approx(math.log1p(math.exp(-1)), abs=1e-12)
    assert logits.grad is not None
    assert logits.grad[0, 2].item() == 0.0


def test_shared_encoder_batch_loss_backpropagates_through_both_sides() -> None:
    class TinyTokenizer:
        padding_side = "right"
        truncation_side = "right"

        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
            ids = torch.zeros((len(texts), 256), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, text in enumerate(texts):
                ids[row, 0] = sum(text.encode("utf-8")) % 30 + 1
                mask[row, 0] = 1
            return {"input_ids": ids, "attention_mask": mask}

    class TinySentenceTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = TinyTokenizer()
            self.embedding = torch.nn.Embedding(31, 4)

        def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {"sentence_embedding": self.embedding(features["input_ids"][:, 0])}

    encoder = object.__new__(SharedMiniLM)
    encoder.device = torch.device("cpu")
    encoder.model = TinySentenceTransformer()
    groups = [
        TrainingGroup("g1", "wireless headphones", ("ear buds", "headphones"), ("phone case",)),
        TrainingGroup("g2", "water bottle", ("steel bottle",), ("glass cup", "mug")),
    ]
    loss = encoder.loss(groups)
    loss.backward()  # type: ignore[no-untyped-call]
    gradients = encoder.model.embedding.weight.grad
    assert gradients is not None
    assert torch.isfinite(loss)
    assert gradients.abs().sum().item() > 0


def test_loss_rejects_groups_without_positive_or_negative_candidates() -> None:
    logits = torch.zeros((1, 2))
    with pytest.raises(ValueError, match="positive"):
        contrastive_group_loss(logits, torch.tensor([[False, False]]), torch.tensor([[True, True]]))
    with pytest.raises(ValueError, match="negative"):
        contrastive_group_loss(logits, torch.tensor([[True, False]]), torch.tensor([[True, False]]))


def test_tokenization_contract_normalizes_and_uses_fixed_right_padding() -> None:
    class FixtureTokenizer:
        padding_side = "left"
        truncation_side = "left"

        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
            assert kwargs == {
                "padding": "max_length",
                "truncation": True,
                "max_length": 256,
                "return_tensors": "pt",
            }
            values = [[ord(char) for char in text[:2]] + [0] * 254 for text in texts]
            masks = [[1] * min(len(text), 2) + [0] * (256 - min(len(text), 2)) for text in texts]
            return {
                "input_ids": torch.tensor(values),
                "attention_mask": torch.tensor(masks),
            }

    tokenizer = FixtureTokenizer()
    features = tokenize_fixed(tokenizer, ["Ａ  X"])
    assert tokenizer.padding_side == tokenizer.truncation_side == "right"
    assert features["input_ids"][0, :3].tolist() == [ord("a"), ord(" "), 0]
    fingerprint = token_id_fingerprint(tokenizer, ["Ａ  X", "a x"])
    assert fingerprint["unique_texts"] == 1
    assert len(fingerprint["sha256"]) == 64


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA hardware unavailable; no GPU allocated"
            ),
        ),
    ],
)
def test_checkpoint_resume_matches_uninterrupted_with_active_dropout(
    tmp_path: Path, device: str
) -> None:
    from atlasrank.training.checkpoints import load_checkpoint
    from atlasrank.training.loop import train_batches

    batches = [torch.tensor([[float(i), 1.0]], device=device) for i in range(1, 5)]
    provenance = {"source": "synthetic-fixture-v1", "seed": 42}

    class DropoutModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.dropout = torch.nn.Dropout(p=0.4)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.dropout(self.linear(values)))

    def initialize() -> tuple[DropoutModel, torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
        torch.manual_seed(42)
        model = DropoutModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
        return model, optimizer, scheduler

    def loss_for(model: DropoutModel, batch: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, model(batch).square().mean())

    full_model, full_optimizer, full_scheduler = initialize()
    initial_parameters = [parameter.detach().clone() for parameter in full_model.parameters()]
    full = train_batches(
        batches,
        model=full_model,
        optimizer=full_optimizer,
        scheduler=full_scheduler,
        loss_for_batch=lambda batch: loss_for(full_model, batch),
        max_global_steps=4,
        checkpoint_directory=tmp_path / "full",
        checkpoint_every=2,
        provenance=provenance,
    )

    partial_model, partial_optimizer, partial_scheduler = initialize()
    partial = train_batches(
        batches,
        model=partial_model,
        optimizer=partial_optimizer,
        scheduler=partial_scheduler,
        loss_for_batch=lambda batch: loss_for(partial_model, batch),
        max_global_steps=2,
        checkpoint_directory=tmp_path / "resumed",
        checkpoint_every=2,
        provenance=provenance,
    )
    assert partial.global_step == partial.sampler_cursor == 2

    resumed_model, resumed_optimizer, resumed_scheduler = initialize()
    step, cursor = load_checkpoint(
        tmp_path / "resumed/step-000002.pt",
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_provenance=provenance,
        map_location=device,
    )
    resumed = train_batches(
        batches,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        loss_for_batch=lambda batch: loss_for(resumed_model, batch),
        global_step=step,
        sampler_cursor=cursor,
        max_global_steps=4,
        checkpoint_directory=tmp_path / "resumed",
        checkpoint_every=2,
        provenance=provenance,
    )
    assert resumed.global_step == resumed.sampler_cursor == 4
    assert full.global_step == 4
    differences = [
        float((left - right).detach().abs().max())
        for left, right in zip(full_model.parameters(), resumed_model.parameters(), strict=True)
    ]
    assert max(differences) <= 1e-7
    assert full_scheduler.last_epoch == resumed_scheduler.last_epoch == 4
    assert any(
        not torch.equal(initial, parameter.detach())
        for initial, parameter in zip(initial_parameters, full_model.parameters(), strict=True)
    )
    for full_parameter, resumed_parameter in zip(
        full_model.parameters(), resumed_model.parameters(), strict=True
    ):
        full_state = full_optimizer.state[full_parameter]
        resumed_state = resumed_optimizer.state[resumed_parameter]
        assert torch.equal(full_state["step"], resumed_state["step"])
        assert torch.allclose(full_state["exp_avg"], resumed_state["exp_avg"], rtol=0, atol=1e-7)
        assert torch.allclose(
            full_state["exp_avg_sq"], resumed_state["exp_avg_sq"], rtol=0, atol=1e-7
        )


def test_checkpoint_rejects_modified_file_and_provenance(tmp_path: Path) -> None:
    from atlasrank.training.checkpoints import load_checkpoint, save_checkpoint

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    provenance = {"source": "test-v1"}
    receipt = save_checkpoint(
        tmp_path,
        global_step=0,
        sampler_cursor=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        provenance=provenance,
    )
    checkpoint = tmp_path / receipt["file"]
    with pytest.raises(FileExistsError):
        save_checkpoint(
            tmp_path,
            global_step=0,
            sampler_cursor=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            provenance=provenance,
        )
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_provenance={"source": "different"},
        )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_provenance=provenance,
        )
