from __future__ import annotations

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_split_manifest_exists():
    manifest_path = ROOT / "research/split_manifest.json"
    assert manifest_path.exists(), "split_manifest.json must exist"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["train_samples"] == 36937
    assert manifest["validation_samples"] == 6519
    assert manifest["official_test_samples"] == 10849
    assert len(manifest["classes"]) == 38


def test_dataset_audit_artifact_verified():
    audit_path = ROOT / "research/artifacts/dataset_split_audit.json"
    assert audit_path.exists(), "dataset_split_audit.json must exist"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["status"] == "verified"
    train_count = audit["samples"]["stratified_train"] if "samples" in audit else audit["train_samples"]
    val_count = audit["samples"]["stratified_validation"] if "samples" in audit else audit["validation_samples"]
    test_count = audit["samples"]["official_test"] if "samples" in audit else audit["official_test_samples"]

    assert train_count == 36937
    assert val_count == 6519
    assert test_count == 10849
    assert audit["num_classes"] == 38


def test_protected_baseline_files_unchanged():
    """Verify that forbidden files are present and not empty."""
    ckpt = ROOT / "research/mobilenetv2_best.pt"
    metrics = ROOT / "research/cnn_test_metrics.json"
    history = ROOT / "research/cnn_history.json"
    manifest = ROOT / "research/split_manifest.json"

    assert ckpt.exists() and ckpt.stat().st_size > 5_000_000
    assert metrics.exists() and metrics.stat().st_size > 100
    assert history.exists() and history.stat().st_size > 1000
    assert manifest.exists() and manifest.stat().st_size > 500
