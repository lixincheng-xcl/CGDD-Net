"""Regression: resuming elsewhere must not silently lose the historical best."""
import json
from pathlib import Path
import pytest
import train


def test_resume_requires_original_output_directory(tmp_path, monkeypatch):
    config=Path(__file__).resolve().parents[1]/'configs/default.json'
    original=tmp_path/'original'; original.mkdir()
    checkpoint=original/'latest.pt'; checkpoint.touch()
    monkeypatch.setattr(train,'load_manifest',lambda *a: [])
    monkeypatch.setattr(train,'validate_splits',lambda *a: {})
    with pytest.raises(ValueError,match='original run directory'):
        train.main(['--config',str(config),'--train-manifest','train.json','--val-manifest','val.json',
                    '--output',str(tmp_path/'new_run'),'--resume',str(checkpoint),'--device','cpu'])
