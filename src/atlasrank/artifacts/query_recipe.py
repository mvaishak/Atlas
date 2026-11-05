"""Explicit portability policy with separate historical and actual recipe identities."""

from typing import Any

from atlasrank.artifacts.dense import digest


def validate_query_recipe(
    actual: dict[str, Any],
    expected_id: str,
    reference: dict[str, Any] | None = None,
) -> None:
    if reference is not None and digest(reference) != expected_id:
        raise ValueError("Reference query recipe digest differs from sealed pin")
    if digest(actual) == expected_id:
        return
    if reference is None:
        raise ValueError("Approved query recipe/runtime differs")
    if {k: v for k, v in actual.items() if k != "runtime_platform"} != {
        k: v for k, v in reference.items() if k != "runtime_platform"
    }:
        raise ValueError("Query recipe scoring/source/dependency pins differ")
    actual_platform = actual.get("runtime_platform", {})
    reference_platform = reference.get("runtime_platform", {})
    if (
        actual_platform.get("machine") != "x86_64"
        or actual_platform.get("machine") != reference_platform.get("machine")
        or set(actual_platform) != {"platform", "machine"}
        or set(reference_platform) != {"platform", "machine"}
        or not str(actual_platform.get("platform", "")).startswith("Linux-")
        or not str(reference_platform.get("platform", "")).startswith("Linux-")
    ):
        raise ValueError("Platform portability requires Linux x86_64")
