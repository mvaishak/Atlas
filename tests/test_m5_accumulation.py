from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from atlasrank.training.loop import train_batches


def mse(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, model(batch).square().mean())


def split(batch: torch.Tensor) -> list[tuple[torch.Tensor, int]]:
    return [(batch[i : i + 8], len(batch[i : i + 8])) for i in range(0, len(batch), 8)]


@pytest.mark.parametrize("count", [16, 12])
def test_weighted_accumulation_matches_effective_batch(count: int) -> None:
    torch.manual_seed(42)
    base = torch.nn.Linear(3, 2)
    accumulated = torch.nn.Linear(3, 2)
    accumulated.load_state_dict(base.state_dict())
    batch = torch.randn(count, 3)
    for model, micro in [(base, None), (accumulated, split)]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        train_batches(
            [batch],
            model=model,
            optimizer=optimizer,
            scheduler=None,
            loss_for_batch=partial(mse, model),
            max_global_steps=1,
            microbatches=micro,
        )
    for a, b in zip(base.parameters(), accumulated.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
        assert a.grad is not None and b.grad is not None
        assert torch.allclose(a.grad, b.grad, atol=1e-6, rtol=1e-5)


def test_durable_update_precedes_guard_and_failure_records_commit(tmp_path: Path) -> None:
    from atlasrank.training.observations import DurableEvents

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "events.jsonl"
    events = DurableEvents(path)

    def guard() -> None:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if records[-1]["phase"] == "completed_update":
            raise RuntimeError("injected pressure")

    with pytest.raises(RuntimeError, match="pressure"):
        train_batches(
            [torch.ones((12, 1))],
            model=model,
            optimizer=optimizer,
            scheduler=None,
            loss_for_batch=partial(mse, model),
            max_global_steps=1,
            microbatches=split,
            diagnostic=events.write,
            guard=guard,
        )
    events.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]["phase"] == "terminal_failure"
    assert records[-1]["global_step"] == records[-1]["sampler_cursor"] == 1
    assert records[-2]["phase"] == "completed_update"
    assert [r["microbatch_groups"] for r in records if r["phase"] == "after_backward"] == [8, 4]
    assert not list(tmp_path.glob("step-*.pt"))


@pytest.mark.parametrize("failure_phase", ["before_forward", "after_forward", "after_backward"])
def test_partial_window_is_not_committed(failure_phase: str) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    before = model.weight.detach().clone()
    records: list[dict[str, Any]] = []

    def event(value: dict[str, Any]) -> None:
        records.append(value)
        if value["phase"] == failure_phase:
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError):
        train_batches(
            [torch.ones((16, 1))],
            model=model,
            optimizer=optimizer,
            scheduler=None,
            loss_for_batch=partial(mse, model),
            max_global_steps=1,
            microbatches=split,
            diagnostic=event,
        )
    assert records[-1]["phase"] == "terminal_failure"
    assert records[-1]["global_step"] == records[-1]["sampler_cursor"] == 0
    assert torch.equal(model.weight, before)


def test_accumulated_dropout_resume_preserves_optimizer_scheduler_and_cursor(
    tmp_path: Path,
) -> None:
    from atlasrank.training.checkpoints import load_checkpoint

    batches = [
        torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 100 for n in [16, 16, 16, 12]
    ]

    def init() -> tuple[torch.nn.Sequential, torch.optim.AdamW, Any]:
        torch.manual_seed(42)
        m = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.4), torch.nn.Linear(4, 2))
        opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1 / (step + 1))
        return m, opt, sch

    def run(
        m: torch.nn.Sequential,
        opt: torch.optim.AdamW,
        sch: Any,
        cap: int,
        directory: Path,
        step: int = 0,
        cursor: int = 0,
    ) -> Any:
        return train_batches(
            batches,
            model=m,
            optimizer=opt,
            scheduler=sch,
            loss_for_batch=lambda b: m(b).square().mean(),
            microbatches=split,
            max_global_steps=cap,
            global_step=step,
            sampler_cursor=cursor,
            checkpoint_directory=directory,
            checkpoint_every=2,
            provenance={"fixture": "dropout-accum-v1"},
        )

    full, fo, fs = init()
    run(full, fo, fs, 4, tmp_path / "full")
    partial, po, ps = init()
    run(partial, po, ps, 2, tmp_path / "partial")
    resumed, ro, rs = init()
    step, cursor = load_checkpoint(
        tmp_path / "partial/step-000002.pt",
        model=resumed,
        optimizer=ro,
        scheduler=rs,
        expected_provenance={"fixture": "dropout-accum-v1"},
    )
    result = run(resumed, ro, rs, 4, tmp_path / "resume", step, cursor)
    assert result.global_step == result.sampler_cursor == rs.last_epoch == fs.last_epoch == 4
    for a, b in zip(full.parameters(), resumed.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-7, rtol=0)
        for key in ["step", "exp_avg", "exp_avg_sq"]:
            assert torch.allclose(fo.state[a][key], ro.state[b][key], atol=1e-7, rtol=0)


def test_nonfinite_gradient_has_json_terminal_evidence() -> None:
    m = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(m.parameters())
    m.weight.register_hook(lambda g: torch.full_like(g, float("inf")))  # type: ignore[no-untyped-call]
    records: list[dict[str, Any]] = []
    with pytest.raises(FloatingPointError):
        train_batches(
            [torch.ones((16, 1))],
            model=m,
            optimizer=opt,
            scheduler=None,
            loss_for_batch=lambda b: m(b).square().mean(),
            microbatches=split,
            diagnostic=records.append,
            max_global_steps=1,
        )
    json.dumps(records, allow_nan=False)
    assert records[-1]["phase"] == "terminal_failure"
    assert records[-1]["gradient_norm_before_clip"] is None
    assert records[-1]["global_step"] == 0 and not opt.state


def test_diagnostic_charge_scope_and_hard_update_cap() -> None:
    from atlasrank.training.budget import remaining_allocation
    from atlasrank.training.configuration import parse_manifest, validate_execution_scope

    manifest = parse_manifest(
        json.loads(Path("configs/m5-colab-pilot.corrected.proposed.json").read_text())
    )
    with pytest.raises(ValueError, match="scope"):
        validate_execution_scope(manifest)
    manifest["prior_m5_gpu_allocation_seconds"] = 1944.7937490940094
    manifest["sampling_and_loss"].update(
        microbatch_size_query_groups=8, gradient_accumulation_steps=2
    )
    manifest["execution_scope"] = {
        "mode": "diagnostic",
        "max_completed_updates": 10,
        "max_active_seconds": 300,
        "max_new_allocation_seconds": 900,
        "automatic_full_pilot_continuation": False,
    }
    assert validate_execution_scope(manifest)["max_completed_updates"] == 10
    assert remaining_allocation(manifest, 0, 200) == 700
    with pytest.raises(TimeoutError):
        remaining_allocation(manifest, 0, 301)
    with pytest.raises(TimeoutError):
        remaining_allocation(manifest, 700, 0)
    manifest["execution_scope"]["max_completed_updates"] = 574
    with pytest.raises(ValueError):
        validate_execution_scope(manifest)


def test_ten_update_cap_never_consumes_eleventh_window() -> None:
    m = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(m.parameters())
    seen: list[int] = []
    batches = list(range(11))

    def loss(index: int) -> torch.Tensor:
        seen.append(index)
        return cast(torch.Tensor, m(torch.ones((1, 1))).square().mean())

    result = train_batches(
        batches, model=m, optimizer=opt, scheduler=None, loss_for_batch=loss, max_global_steps=10
    )
    assert result.global_step == result.sampler_cursor == 10
    assert seen == list(range(10)) and result.stopped_at_step_cap


def test_actual_multi_positive_objective_accumulates_with_unequal_candidates() -> None:
    from atlasrank.training.dual_encoder import batch_masks, contrastive_group_loss

    torch.manual_seed(42)
    a = torch.nn.Linear(1, 6)
    b = torch.nn.Linear(1, 6)
    b.load_state_dict(a.state_dict())
    inputs = torch.arange(12, dtype=torch.float32).reshape(-1, 1) / 10

    def objective(model: torch.nn.Module, values: torch.Tensor) -> torch.Tensor:
        counts = [2 if int(round(float(v) * 10)) % 2 else 1 for v in values[:, 0]]
        pos, valid = batch_masks(counts, [4 if c == 2 else 1 for c in counts], device="cpu")
        return contrastive_group_loss(model(values), pos, valid)

    for model, parts in [(a, None), (b, split)]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        train_batches(
            [inputs],
            model=model,
            optimizer=optimizer,
            scheduler=None,
            loss_for_batch=partial(objective, model),
            microbatches=parts,
            max_global_steps=1,
        )
    for left, right in zip(a.parameters(), b.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        assert torch.allclose(left.grad, right.grad, atol=1e-6, rtol=1e-5)
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)


def test_evidence_retains_no_autograd_tensors_and_close_flushes(tmp_path: Path) -> None:
    from atlasrank.training.observations import DurableEvents

    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "durable.jsonl"
    writer = DurableEvents(path)
    events: list[dict[str, Any]] = []

    def observe(event: dict[str, Any]) -> None:
        assert not any(isinstance(value, torch.Tensor) for value in event.values())
        events.append(event)
        writer.write(event)

    try:
        train_batches(
            [torch.ones((12, 1))],
            model=model,
            optimizer=opt,
            scheduler=None,
            loss_for_batch=partial(mse, model),
            microbatches=split,
            max_global_steps=1,
            diagnostic=observe,
            batch_details=lambda batch: {"text_count": len(batch)},
        )
    finally:
        writer.close()
    assert writer.stream.closed
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == events
    assert records[-1]["phase"] == "completed_update"
    assert records[-1]["loss"] is not None and records[-1]["gradient_norm_before_clip"] is not None


@pytest.mark.parametrize(
    "phase", ["before_forward", "after_forward", "after_backward", "completed_update"]
)
def test_measured_low_free_memory_stops_at_phase(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from atlasrank.training.observations import require_gpu_memory

    current = ["initial"]
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: ((7 if current[0] == phase else 9) * 1024**3, 24 * 1024**3),
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 3 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 16 * 1024**3)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 4 * 1024**3)
    m = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(m.parameters())

    def record(value: dict[str, Any]) -> None:
        current[0] = value["phase"]
        records.append(value)

    with pytest.raises(RuntimeError, match="free VRAM"):
        train_batches(
            [torch.ones((16, 1))],
            model=m,
            optimizer=opt,
            scheduler=None,
            loss_for_batch=partial(mse, m),
            microbatches=split,
            max_global_steps=1,
            diagnostic=record,
            guard=lambda: require_gpu_memory(
                minimum_free=8 * 1024**3, maximum_allocated=10 * 1024**3
            ),
        )
    assert records[-1]["phase"] == "terminal_failure"
    assert records[-1]["global_step"] == (1 if phase == "completed_update" else 0)


def test_checkpoint_failure_emits_committed_terminal_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("injected checkpoint failure")

    monkeypatch.setattr("atlasrank.training.loop.save_checkpoint", broken)
    m = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(m.parameters())
    records: list[dict[str, Any]] = []
    with pytest.raises(OSError, match="checkpoint"):
        train_batches(
            [torch.ones((12, 1))],
            model=m,
            optimizer=opt,
            scheduler=None,
            loss_for_batch=partial(mse, m),
            microbatches=split,
            max_global_steps=1,
            checkpoint_directory=tmp_path,
            checkpoint_every=100,
            provenance={"fixture": 1},
            diagnostic=records.append,
        )
    assert records[-1]["phase"] == "terminal_failure"
    assert records[-1]["global_step"] == records[-1]["sampler_cursor"] == 1


def test_exact_unapproved_diagnostic_proposal_parses() -> None:
    from atlasrank.training.configuration import parse_manifest, validate_execution_scope

    raw = json.loads(Path("configs/m5-colab-diagnostic.proposed.json").read_text())
    parsed = parse_manifest(raw)
    assert validate_execution_scope(parsed)["max_completed_updates"] == 10
    assert parsed["max_optimizer_steps"] == 574 and parsed["batch_size"] == 16
    assert parsed["prior_m5_gpu_allocation_seconds"] == 1944.7937490940094
    assert not raw["execution_approved"] and not raw["runnable"] and not raw["storage_approved"]
    assert raw["operator_owner_token"] is None


@pytest.mark.parametrize(
    "runtime,expected", [(600, "completed"), (701, "allocation_limit_exceeded")]
)
def test_setup_is_charged_once_at_session_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: int, expected: str
) -> None:
    from atlasrank.training.budget import AllocationSession, finish_session

    session_id = "a" * 32
    active = tmp_path / "ACTIVE.json"
    active.write_text(json.dumps({"session_id": session_id}))
    monkeypatch.setattr("atlasrank.training.budget.time.monotonic", lambda: float(100 + runtime))
    session = AllocationSession(
        tmp_path,
        tmp_path / (session_id + ".start.json"),
        session_id,
        0.0,
        100.0,
        200.0,
        700.0,
        "manifest-hash",
        active,
    )
    result = finish_session(session, "completed")
    assert result["elapsed_seconds"] == 200 + runtime
    assert result["status"] == expected
