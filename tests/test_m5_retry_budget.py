import json
from pathlib import Path

import pytest

from atlasrank.training.budget import remaining_allocation
from atlasrank.training.configuration import parse_manifest, validate_execution_scope


def test_retry_allows_1000_setup_seconds_but_preserves_execution_reserves() -> None:
    manifest = parse_manifest(
        json.loads(Path("configs/m5-colab-diagnostic.proposed.json").read_text())
    )
    manifest["execution_scope"]["max_new_allocation_seconds"] = 1600
    validate_execution_scope(manifest)
    assert remaining_allocation(manifest, 0, 1000) == 600
    with pytest.raises(TimeoutError):
        remaining_allocation(manifest, 0, 1000.01)
    for field, value in [
        ("max_completed_updates", 11),
        ("max_active_seconds", 1000),
        ("max_new_allocation_seconds", 1601),
    ]:
        changed = dict(
            manifest, execution_scope=dict(manifest["execution_scope"], **{field: value})
        )
        with pytest.raises(ValueError):
            validate_execution_scope(changed)
