import json
import pytest
from pathlib import Path
from research.utils.io import load_json_safe, save_json

def test_load_json_safe_success(tmp_path):
    data = {"key": "value"}
    file_path = tmp_path / "test.json"
    save_json(file_path, data)
    
    assert load_json_safe(file_path) == data

def test_load_json_safe_file_not_found():
    assert load_json_safe("non_existent.json", default={}) == {}
    assert load_json_safe("non_existent.json") is None

def test_load_json_safe_invalid_json(tmp_path):
    file_path = tmp_path / "invalid.json"
    file_path.write_text("invalid json")
    
    assert load_json_safe(file_path, default={}) == {}
    assert load_json_safe(file_path) is None
