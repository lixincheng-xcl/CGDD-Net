"""Static validation of Kaggle notebook; never runs or claims GPU experiments."""
from __future__ import annotations
import ast
import json
from pathlib import Path


def test_notebook_imports_and_cells_are_valid_python():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "CGDDNet_PR_Kaggle_First_Batch.ipynb"
    book = json.loads(path.read_text(encoding="utf-8"))
    assert book["nbformat"] == 4
    cells = book["cells"]
    assert len(cells) >= 15
    code = []
    for cell in cells:
        if cell["cell_type"] == "code":
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            ast.parse(source)
            code.append(source)
    joined = "\n".join(code)
    assert len(code) >= 8
    assert "experiments.kaggle_data" in joined
    assert "experiments.train_controlled" in joined
    assert "--source-folder" in joined
    assert "PINNED_COMMIT" in joined
    assert "--epochs-override" in joined
    assert "RUN_MODE = \"first_run\"" in joined
    assert "DATASET = \"CHASEDB1\"" in joined