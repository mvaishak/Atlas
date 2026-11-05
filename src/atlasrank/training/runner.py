"""Run the frozen one-pass training plan from a verified private input bundle."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from atlasrank.data.common import sha256
from atlasrank.retrieval.colab_environment import (
    inspect_runtime,
    validate_installed,
    validate_runtime,
)
from atlasrank.retrieval.colab_transport import once, publish
from atlasrank.training.budget import begin_session, finish_session
from atlasrank.training.checkpoints import load_checkpoint, save_checkpoint
from atlasrank.training.configuration import parse_manifest, validate_execution_scope
from atlasrank.training.loop import TrainProgress, train_batches
from atlasrank.training.model import SharedMiniLM, TrainingGroup
from atlasrank.training.observations import DurableEvents, gpu_memory, require_gpu_memory
from atlasrank.training.tokenization import token_id_fingerprint


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        tensor = value.detach().cpu().contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("Sanity check found non-finite model weights")
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_sanity(model: torch.nn.Module, initial: str, event: dict[str, Any]) -> None:
    if not all(math.isfinite(float(event[k])) for k in ("loss", "gradient_norm_before_clip")):
        raise RuntimeError("Sanity check found non-finite loss or gradient")
    if event["cuda_free_bytes"] < 8 * 1024**3:
        raise RuntimeError("Sanity check free VRAM floor failed")
    if max(event["cuda_allocated_bytes"], event["cumulative_peak_allocated_bytes"]) > 10 * 1024**3:
        raise RuntimeError("Sanity check peak VRAM cap failed")
    if _model_digest(model) == initial:
        raise RuntimeError("Sanity check found unchanged model weights")


def _group_batches(records: list[dict[str, Any]], batch_size: int) -> list[list[TrainingGroup]]:
    groups = [
        TrainingGroup(
            group_key=str(row["group_key"]),
            query=str(row["query"]),
            positive_titles=tuple(str(item["title"]) for item in row["positives"]),
            negative_titles=tuple(str(item["title"]) for item in row["negatives"]),
        )
        for row in records
    ]
    return [groups[offset : offset + batch_size] for offset in range(0, len(groups), batch_size)]


def run_training(
    *,
    input_archive: Path,
    checkpoint_directory: Path,
    manifest_path: Path,
    model_directory: Path,
    input_root: Path,
    persistent_directory: Path,
    allocation_started_utc: float,
    recover_interrupted: bool = False,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    manifest = parse_manifest(json.loads(manifest_path.read_text()))
    if manifest.get("execution_approved") is not True:
        raise PermissionError("Training manifest is unapproved; GPU training is blocked")
    scope = validate_execution_scope(manifest)
    if resume_checkpoint is not None and scope["mode"] == "diagnostic":
        raise PermissionError("Diagnostic is a single attempt; no implicit resume or continuation")
    session = begin_session(
        manifest_path,
        persistent_directory,
        allocation_started_utc=allocation_started_utc,
        recover_interrupted=recover_interrupted,
    )
    if not checkpoint_directory.resolve().is_relative_to(persistent_directory.resolve()):
        raise PermissionError("M5 checkpoints must remain inside the approved private Drive folder")
    if manifest.get("input_archive_sha256") != sha256(input_archive):
        raise ValueError("Training export archive checksum differs from approved manifest")
    if input_archive.stat().st_size != manifest.get("input_archive_bytes") or (
        input_archive.stat().st_size > int(manifest["upload_limit_bytes"])
    ):
        raise ValueError("Training export size differs from approved transfer budget")
    with zipfile.ZipFile(input_archive) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Corrupt training export member: {bad_member}")
        export_metadata = json.loads(archive.read("EXPORT.json"))
        plan_bytes = archive.read("sampling-plan.json")
        plan = json.loads(plan_bytes)
        records = [json.loads(line) for line in archive.read("train-groups.jsonl").splitlines()]
    if (
        export_metadata["train_manifest_sha256"] != manifest["train_manifest_sha256"]
        or export_metadata["catalog_sha256"] != manifest["catalog_sha256"]
        or export_metadata["sampling_plan_sha256"] != manifest["sampling_plan_sha256"]
        or hashlib.sha256(plan_bytes).hexdigest() != manifest["sampling_plan_sha256"]
        or export_metadata["evaluation_contract_sha256"] != manifest["evaluation_contract_sha256"]
        or export_metadata["source_sha256"] != manifest["source_sha256"]
        or export_metadata["m4_reference"]["m3_bm25_bundle_id"] != manifest["m3_bm25_reference_id"]
        or export_metadata["m4_reference"]["m4_dense_bundle_id"]
        != manifest["m4_dense_reference_id"]
        or export_metadata["m4_reference"]["m3_reference_sha256"] != manifest["m3_reference_sha256"]
        or export_metadata["m4_reference"]["m4_report_archive_sha256"]
        != manifest["m4_report_archive_sha256"]
        or export_metadata["m4_reference"]["rrf_k"] != manifest["rrf_k"]
        or export_metadata["m4_validation_sha256"] != manifest["m4_validation_sha256"]
        or export_metadata["dense_cpu_config_sha256"] != manifest["dense_cpu_config_sha256"]
        or export_metadata["train_rows"] != manifest["train_rows"]
        or export_metadata["label_counts"] != manifest["label_counts"]
        or export_metadata["selected_product_identity_count"]
        != manifest["selected_product_identity_count"]
        or export_metadata["selected_positive_pair_count"]
        != manifest["selected_positive_pair_count"]
        or export_metadata["selected_negative_pair_count"]
        != manifest["selected_negative_pair_count"]
        or export_metadata["token_id_fingerprint"] != manifest["token_id_fingerprint"]
        or plan["usable_group_count"] != manifest["usable_group_count"]
        or len(records) != plan["usable_group_count"]
    ):
        raise ValueError("Training data provenance differs from approved manifest")
    for relative, expected_hash in export_metadata["source_sha256"].items():
        base = input_root if relative.startswith("environments/") else input_root / "source"
        candidate = base / relative
        if not candidate.is_file() or sha256(candidate) != expected_hash:
            raise ValueError(f"Reviewed training source/environment differs: {relative}")
    environment_path = input_root / "environments/colab/environment.json"
    environment: dict[str, Any] = json.loads(environment_path.read_text())
    if sha256(environment_path) != manifest["environment_sha256"]:
        raise ValueError("Colab dependency environment manifest differs from approval")
    runtime_info = inspect_runtime()
    validate_runtime(runtime_info, environment)
    installed_versions = validate_installed(environment)
    for relative, expected_hash in export_metadata["model_file_sha256"].items():
        candidate = model_directory / relative
        if not candidate.is_file() or sha256(candidate) != expected_hash:
            raise ValueError(f"Pinned base model file differs: {relative}")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("M5 pilot requires exactly one available CUDA device")
    if torch.cuda.get_device_name(0) != manifest["gpu_name"]:
        raise RuntimeError("M5 pilot requires the manifest's exact GPU model")
    import psutil

    memory = psutil.virtual_memory()
    if memory.available < int(manifest["available_ram_start_bytes"]):
        raise RuntimeError("Available host RAM is below the pilot launch gate")
    if psutil.Process().memory_info().rss > int(manifest["rss_limit_bytes"]):
        raise RuntimeError("Worker RSS exceeds the pilot limit before model loading")
    require_gpu_memory(
        minimum_free=int(manifest["minimum_free_vram_bytes"]),
        maximum_allocated=int(manifest["vram_limit_bytes"]),
    )
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()

    started = time.monotonic()
    model = SharedMiniLM(str(model_directory), "cuda:0")
    model_load_seconds = time.monotonic() - started
    tokenization_started = time.monotonic()
    text_inputs = [str(row["query"]) for row in records]
    text_inputs.extend(
        str(item["title"]) for row in records for item in (*row["positives"], *row["negatives"])
    )
    actual_token_fingerprint = token_id_fingerprint(model.model.tokenizer, text_inputs)
    if actual_token_fingerprint != export_metadata["token_id_fingerprint"]:
        raise ValueError("Colab tokenizer IDs differ from the reviewed training export")
    tokenization_seconds = time.monotonic() - tokenization_started
    model.model.train()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=2e-5, weight_decay=0.01)
    batches = _group_batches(records, int(manifest["batch_size"]))
    total_steps = min(len(batches), int(manifest["max_optimizer_steps"]))
    warmup_steps = max(1, math.ceil(total_steps * 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        remaining = max(0, total_steps - step)
        return remaining / max(1, total_steps - warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if total_steps != 574 or len(batches) != 574:
        raise ValueError("Approved full pilot schedule must contain exactly 574 batches")
    provenance = {
        "schema_version": 1,
        "export_sha256": manifest["input_archive_sha256"],
        "pilot_manifest_sha256": sha256(manifest_path),
        "train_manifest_sha256": manifest["train_manifest_sha256"],
        "catalog_sha256": manifest["catalog_sha256"],
        "sampling_plan_sha256": manifest["sampling_plan_sha256"],
        "model_repo": export_metadata["model_repo"],
        "model_revision": export_metadata["model_revision"],
        "model_file_sha256": export_metadata["model_file_sha256"],
        "temperature": manifest["temperature"],
        "batch_size": manifest["batch_size"],
        "optimizer": "AdamW",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "seed": 42,
        "precision": "float32",
        "gradient_clip_norm": float(manifest["gradient_clip_norm"]),
        "max_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "scheduler": "linear_warmup_then_linear_decay",
        "loss": "equal_query_mean_logsumexp_summed_positive_softmax",
        "microbatch_size_query_groups": 8,
        "gradient_accumulation_steps": 2,
        "execution_scope": scope,
        "source_sha256": export_metadata["source_sha256"],
        "environment_sha256": manifest["environment_sha256"],
        "evaluation_contract_sha256": manifest["evaluation_contract_sha256"],
        "m4_dense_reference_id": manifest["m4_dense_reference_id"],
        "m4_hybrid_reference_id": manifest["m4_hybrid_reference_id"],
        "m3_bm25_reference_id": manifest["m3_bm25_reference_id"],
        "python_version": __import__("sys").version,
        "numpy_version": np.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    global_step = 0
    cursor = 0
    initial_model_digest = _model_digest(model.model)
    sanity_passed = [False]
    sanity_checkpoint_seconds = [0.0]
    session_checkpoints = checkpoint_directory / session.session_id
    if resume_checkpoint is not None:
        global_step, cursor = load_checkpoint(
            resume_checkpoint,
            model=model.model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_provenance=provenance,
            map_location="cuda:0",
        )
        if cursor != global_step or cursor > len(batches):
            raise ValueError("Checkpoint step and consumed sampler cursor are inconsistent")
        if global_step >= total_steps:
            raise PermissionError("The approved one-pass training schedule is already complete")
        sanity_passed[0] = global_step >= 10

    next_storage_check = [0.0]

    def guard() -> None:
        elapsed = time.monotonic() - started
        allocated_elapsed = time.monotonic() - session.session_started_monotonic
        if elapsed >= min(
            int(manifest["max_active_training_seconds"]), int(scope["max_active_seconds"])
        ):
            raise TimeoutError("M5 approved active-run cap reached")
        if (
            allocated_elapsed + int(manifest["finalization_reserve_seconds"])
            >= session.remaining_seconds
        ):
            raise TimeoutError("M5 allocation reached its training/finalization stop margin")
        current_memory = psutil.virtual_memory()
        if psutil.Process().memory_info().rss > int(manifest["rss_limit_bytes"]):
            raise RuntimeError("M5 worker RSS cap reached")
        if current_memory.available < 4 * 1024**3:
            raise RuntimeError("M5 available host RAM fell below the 4 GiB stop floor")
        if time.monotonic() >= next_storage_check[0]:
            if _directory_size(persistent_directory) > int(manifest["persistent_limit_bytes"]):
                raise RuntimeError("M5 persistent checkpoint storage cap reached")
            work_directory = Path("/content/atlasrank-m5-work")
            scratch_bytes = _directory_size(work_directory)
            if not input_archive.resolve().is_relative_to(work_directory.resolve()):
                scratch_bytes += input_archive.stat().st_size
            if scratch_bytes > int(manifest["scratch_limit_bytes"]):
                raise RuntimeError("M5 Colab scratch cap reached")
            next_storage_check[0] = time.monotonic() + 10
        require_gpu_memory(
            minimum_free=int(manifest["minimum_free_vram_bytes"]),
            maximum_allocated=int(manifest["vram_limit_bytes"]),
        )

    training_started = time.monotonic()
    progress_directory = persistent_directory / "logs"
    progress_directory.mkdir(parents=True, exist_ok=True)
    progress_path = progress_directory / f"{session.session_id}.progress.jsonl"
    writer = DurableEvents(progress_path)
    cumulative_peak = [0]
    logging_seconds = [0.0]

    def diagnostic(event: dict[str, Any]) -> None:
        logging_started = time.monotonic()
        torch.cuda.synchronize()
        memory_snapshot = gpu_memory()
        cumulative_peak[0] = max(cumulative_peak[0], memory_snapshot["cuda_peak_allocated_bytes"])
        record = {
            **event,
            **memory_snapshot,
            "session_id": session.session_id,
            "event": "m5_training_diagnostic",
            "cumulative_peak_allocated_bytes": cumulative_peak[0],
            "host_available_ram_bytes": psutil.virtual_memory().available,
            "worker_rss_bytes": psutil.Process().memory_info().rss,
            "allocation_elapsed_seconds": session.allocation_elapsed_before_start
            + time.monotonic()
            - session.session_started_monotonic,
            "previous_runs_charged_seconds": manifest["prior_m5_gpu_allocation_seconds"],
            "logging_seconds_so_far": logging_seconds[0],
        }
        writer.write(record)
        logging.info(json.dumps(record, sort_keys=True, allow_nan=False))
        if scope["mode"] == "sanity_then_training":
            if event["phase"] == "window_start" and event["global_step"] >= 10:
                if not sanity_passed[0]:
                    raise RuntimeError("Cannot continue before ten-update sanity check passes")
            if event["phase"] == "completed_update" and event["global_step"] == 10:
                guard()
                _validate_sanity(model.model, initial_model_digest, record)
                checkpoint_started = time.monotonic()
                receipt = save_checkpoint(
                    session_checkpoints,
                    global_step=10,
                    sampler_cursor=10,
                    model=model.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    provenance=provenance,
                )
                sanity_checkpoint_seconds[0] = time.monotonic() - checkpoint_started
                sanity_passed[0] = True
                writer.write(
                    {
                        "event": "sanity_passed",
                        "global_step": 10,
                        "weights_updated": True,
                        "checkpoint_receipt": receipt,
                        "checkpoint_seconds": sanity_checkpoint_seconds[0],
                    }
                )
                logging.info("Ten-update GPU sanity passed; continuing frozen 574-update schedule")
        if event["phase"] == "window_start":
            torch.cuda.reset_peak_memory_stats()
        logging_seconds[0] += time.monotonic() - logging_started

    try:
        progress: TrainProgress = train_batches(
            batches,
            model=model.model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_for_batch=lambda batch: model.loss(batch, float(manifest["temperature"])),
            microbatches=lambda batch: [
                (batch[i : i + 8], len(batch[i : i + 8])) for i in range(0, len(batch), 8)
            ],
            batch_details=lambda batch: {
                "query_texts": len(batch),
                "product_texts": sum(
                    len(g.positive_titles) + len(g.negative_titles) for g in batch
                ),
                "text_count": sum(
                    1 + len(g.positive_titles) + len(g.negative_titles) for g in batch
                ),
            },
            global_step=global_step,
            sampler_cursor=cursor,
            max_global_steps=int(scope["max_completed_updates"]),
            grad_clip_norm=float(manifest["gradient_clip_norm"]),
            checkpoint_directory=session_checkpoints,
            checkpoint_every=int(manifest["checkpoint_interval_steps"]),
            provenance=provenance,
            guard=guard,
            diagnostic=diagnostic,
        )
    except BaseException:
        # train_batches durably records the last committed cursor before propagating.
        finish_session(session, "failed")
        raise
    finally:
        writer.close()
    torch.cuda.synchronize()
    training_seconds = time.monotonic() - training_started
    elapsed = time.monotonic() - started
    allocated_elapsed = session.allocation_elapsed_before_start + (
        time.monotonic() - session.session_started_monotonic
    )
    peak_allocated = max(cumulative_peak[0], torch.cuda.max_memory_allocated())
    if peak_allocated > int(manifest["vram_limit_bytes"]):
        raise RuntimeError("M5 peak allocated VRAM cap exceeded")
    completed = progress.sampler_cursor == len(batches)
    return_directory = Path("/content/atlasrank-work/return") / session.session_id
    return_directory.mkdir(parents=True, exist_ok=False)
    save_seconds = 0.0
    if completed:
        save_started = time.monotonic()
        model.model.save_pretrained(return_directory / "model", safe_serialization=True)
        save_seconds = time.monotonic() - save_started
    (return_directory / "progress.jsonl").write_bytes(progress_path.read_bytes())
    report = {
        "schema_version": 1,
        "status": "completed" if completed else "diagnostic_completed",
        "global_step": progress.global_step,
        "sampler_cursor": progress.sampler_cursor,
        "total_batches": len(batches),
        "mean_loss_this_session": sum(progress.losses) / len(progress.losses)
        if progress.losses
        else None,
        "session_seconds": elapsed,
        "allocation_elapsed_seconds": allocated_elapsed,
        "cuda_peak_allocated_bytes": peak_allocated,
        "provenance": provenance,
        "checkpoint_files": sorted(
            str(path.relative_to(checkpoint_directory))
            for path in checkpoint_directory.rglob("step-*.pt")
        ),
        "archive_sha256": sha256(input_archive),
        "model_load_seconds": model_load_seconds,
        "tokenization_validation_seconds": tokenization_seconds,
        "training_seconds": training_seconds,
        "checkpoint_seconds": progress.checkpoint_seconds,
        "sanity_passed": sanity_passed[0],
        "sanity_checkpoint_seconds": sanity_checkpoint_seconds[0],
        "diagnostic_logging_seconds": logging_seconds[0],
        "diagnostic_scope": scope,
        "model_export_seconds": save_seconds,
        "cuda_device": torch.cuda.get_device_name(0),
        "installed_versions": installed_versions,
        "runtime_info": runtime_info,
        "host_available_ram_bytes_at_end": psutil.virtual_memory().available,
        "worker_rss_bytes_at_end": psutil.Process().memory_info().rss,
        "session_id": session.session_id,
        "allocation_start_utc": allocation_started_utc,
        "cumulative_budget_seconds": manifest["gpu_allocation_seconds"],
    }
    (return_directory / "training-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    if _directory_size(persistent_directory) + _directory_size(
        return_directory
    ) + 1024 * 1024 > int(manifest["persistent_limit_bytes"]):
        raise RuntimeError("M5 persistent storage cap would be exceeded by the return bundle")
    output_receipt = publish(
        return_directory,
        persistent_directory / "returns",
        f"m5-pilot-{session.session_id}",
        int(manifest["return_limit_bytes"]),
    )
    if _directory_size(persistent_directory) > int(manifest["persistent_limit_bytes"]):
        raise RuntimeError("M5 persistent storage cap would be exceeded by the return bundle")
    session_record = finish_session(session, "completed" if completed else "diagnostic_completed")
    if session_record["status"] == "allocation_limit_exceeded":
        report["status"] = "allocation_limit_exceeded"
    run_record = {
        **report,
        "allocation_elapsed_seconds": session_record["elapsed_seconds"],
        "return_receipt": output_receipt,
        "session_record": session_record,
    }
    once(persistent_directory / f"{session.session_id}.run.json", run_record)
    if session_record["status"] == "allocation_limit_exceeded":
        raise TimeoutError("M5 allocation duration exceeded its remaining cumulative budget")
    return run_record


def _directory_size(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
