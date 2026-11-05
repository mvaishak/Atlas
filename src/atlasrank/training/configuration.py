"""Resolve the reviewed nested manifest; never use root settings as defaults."""

from __future__ import annotations

from typing import Any

PATHS = {
    "gpu_allocation_seconds": ("resource_limits", "gpu_allocation_cumulative_seconds"),
    "max_active_training_seconds": ("resource_limits", "max_active_runner_seconds"),
    "finalization_reserve_seconds": ("resource_limits", "finalization_reserve_seconds"),
    "max_optimizer_steps": ("resource_limits", "max_optimizer_steps"),
    "vram_limit_bytes": ("resource_limits", "peak_allocated_vram_bytes"),
    "minimum_free_vram_bytes": ("resource_limits", "minimum_free_vram_bytes"),
    "available_ram_start_bytes": ("resource_limits", "minimum_available_host_ram_at_start_bytes"),
    "rss_limit_bytes": ("resource_limits", "worker_rss_bytes"),
    "scratch_limit_bytes": ("resource_limits", "colab_scratch_bytes"),
    "persistent_limit_bytes": ("resource_limits", "persistent_drive_bytes"),
    "upload_limit_bytes": ("resource_limits", "upload_bytes"),
    "return_limit_bytes": ("resource_limits", "return_bytes"),
    "prior_m5_gpu_allocation_seconds": ("resource_limits", "prior_m5_allocation_seconds"),
    "batch_size": ("sampling_and_loss", "batch_size_query_groups"),
    "temperature": ("sampling_and_loss", "temperature"),
    "gradient_clip_norm": ("sampling_and_loss", "gradient_clip_norm"),
    "seed": ("sampling_and_loss", "seed"),
    "precision": ("sampling_and_loss", "precision"),
    "tf32": ("sampling_and_loss", "tf32"),
    "autocast": ("sampling_and_loss", "autocast"),
    "max_positives": ("sampling_and_loss", "max_positive_products_per_group"),
    "max_negatives": ("sampling_and_loss", "max_admissible_I_negatives_per_group"),
    "usable_group_count": ("sampling_and_loss", "one_pass_groups"),
    "checkpoint_interval_steps": ("checkpointing", "interval_steps"),
    "model_repo": ("model", "repo"),
    "model_revision": ("model", "revision"),
    "gpu_name": ("execution_environment", "gpu_name"),
    "train_manifest_sha256": ("training_data", "train_manifest_sha256"),
    "catalog_sha256": ("training_data", "catalog_sha256"),
    "sampling_plan_sha256": ("training_data", "sampling_plan_sha256"),
    "train_rows": ("training_data", "train_rows"),
    "label_counts": ("training_data", "train_label_counts"),
    "selected_product_identity_count": ("training_data", "sampled_product_identities"),
    "selected_positive_pair_count": ("training_data", "sampled_positive_pairs"),
    "selected_negative_pair_count": ("training_data", "sampled_negative_pairs"),
    "token_id_fingerprint": ("training_data", "token_id_fingerprint"),
    "evaluation_contract_sha256": ("m4_baselines", "evaluation_contract_sha256"),
    "m3_bm25_reference_id": ("m4_baselines", "m3_bm25_bundle_id"),
    "m4_dense_reference_id": ("m4_baselines", "m4_dense_bundle_id"),
    "m3_reference_sha256": ("m4_baselines", "m3_reference_sha256"),
    "m4_report_archive_sha256": ("m4_baselines", "m4_report_archive_sha256"),
    "rrf_k": ("m4_baselines", "rrf_k"),
    "m4_validation_sha256": ("m4_baselines", "m4_validation_sha256"),
    "dense_cpu_config_sha256": ("m4_baselines", "dense_cpu_config_sha256"),
    "environment_sha256": ("provenance", "colab_environment_sha256"),
}
FROZEN = {
    "gpu_allocation_seconds": 3600,
    "max_active_training_seconds": 2700,
    "finalization_reserve_seconds": 300,
    "max_optimizer_steps": 574,
    "vram_limit_bytes": 10737418240,
    "minimum_free_vram_bytes": 8589934592,
    "available_ram_start_bytes": 17179869184,
    "rss_limit_bytes": 12884901888,
    "scratch_limit_bytes": 8589934592,
    "persistent_limit_bytes": 3221225472,
    "upload_limit_bytes": 536870912,
    "return_limit_bytes": 536870912,
    "batch_size": 16,
    "temperature": 0.05,
    "gradient_clip_norm": 1.0,
    "seed": 42,
    "precision": "float32",
    "tf32": False,
    "autocast": False,
    "max_positives": 2,
    "max_negatives": 4,
    "usable_group_count": 9180,
    "checkpoint_interval_steps": 100,
    "model_repo": "sentence-transformers/all-MiniLM-L6-v2",
    "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}
CONTRACT = {
    "sampling_and_loss.optimizer": "AdamW",
    "sampling_and_loss.learning_rate": 2e-05,
    "sampling_and_loss.weight_decay": 0.01,
    "sampling_and_loss.warmup_fraction": 0.05,
    "sampling_and_loss.linear_decay": True,
    "sampling_and_loss.loss": (
        "per-query logsumexp(all valid positive and negative logits) minus "
        "logsumexp(all positive logits); equal-weight mean over query groups"
    ),
    "sampling_and_loss.cross_query_in_batch_negatives": False,
    "sampling_and_loss.random_catalog_negatives": False,
    "sampling_and_loss.development_label_supervision": False,
    "sampling_and_loss.excluded_labels": ["S", "C", "unknown"],
    "sampling_and_loss.negative_admissibility": (
        "same training query group, explicit I only; remove any product "
        "with E anywhere in the train manifest"
    ),
    "sampling_and_loss.without_replacement": True,
}


def parse_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(raw)
    for name, (section, key) in PATHS.items():
        if not isinstance(raw.get(section), dict) or key not in raw[section]:
            raise ValueError(f"Missing required nested configuration: {section}.{key}")
        value = raw[section][key]
        if name in raw and raw[name] != value:
            raise ValueError(f"Conflicting root/nested configuration: {name}")
        if name in FROZEN and value != FROZEN[name]:
            raise ValueError(f"Unapproved configuration: {section}.{key}")
        resolved[name] = value
    for path, expected in CONTRACT.items():
        section, key = path.split(".")
        if raw[section].get(key) != expected:
            raise ValueError(f"Missing or unapproved configuration: {path}")
    if raw.get("kind") != "m5_colab_training_pilot":
        raise ValueError("Invalid pilot kind")
    objective = "equal_query_mean_logsumexp_summed_positive_softmax"
    if "loss" in raw and raw["loss"] != objective:
        raise ValueError("Conflicting root/nested configuration: loss")
    if raw["sampling_and_loss"].get("maximum_optimizer_steps") != resolved["max_optimizer_steps"]:
        raise ValueError("Conflicting nested configuration: maximum_optimizer_steps")
    resolved["loss"] = objective
    resolved["m4_hybrid_reference_id"] = (
        f"rrf60:{resolved['m3_bm25_reference_id']}:{resolved['m4_dense_reference_id']}"
    )
    return resolved


def validate_execution_scope(manifest: dict[str, Any]) -> dict[str, Any]:
    """New source permits only the separately approved bounded diagnostic."""
    scope = manifest.get("execution_scope")
    required = {
        "mode": "diagnostic",
        "max_completed_updates": 10,
        "max_active_seconds": 300,
        "max_new_allocation_seconds": 900,
        "automatic_full_pilot_continuation": False,
    }
    retry = {**required, "max_new_allocation_seconds": 1600}
    training = {
        "mode": "sanity_then_training",
        "max_completed_updates": 574,
        "max_active_seconds": 2700,
        "max_new_allocation_seconds": 3600,
        "automatic_full_pilot_continuation": True,
    }
    if scope not in (required, retry, training):
        raise ValueError("Missing or unapproved diagnostic execution scope")
    sampling = manifest["sampling_and_loss"]
    if (
        sampling.get("microbatch_size_query_groups") != 8
        or sampling.get("gradient_accumulation_steps") != 2
    ):
        raise ValueError("Diagnostic requires microbatch 8, accumulation 2")
    expected_prior = 0 if scope == training else 1944.7937490940094
    if manifest["prior_m5_gpu_allocation_seconds"] != expected_prior:
        raise ValueError("Prior M5 allocation charge differs from verified release")
    return dict(scope)
