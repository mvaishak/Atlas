"""Keep maintained notebook templates data-free and syntactically valid."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda path: path.name)
def test_notebook_template_has_unexecuted_valid_code(path: Path) -> None:
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        compile("".join(cell["source"]), f"{path.name}:cell{index}", "exec")
