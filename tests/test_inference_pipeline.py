from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from research.inference.predict import PredictionPipeline

ROOT = Path(__file__).resolve().parents[1]


def test_inference_pipeline_execution():
    pipeline = PredictionPipeline()
    assert len(pipeline.classes) == 38

    # Create temporary dummy image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        img = Image.new("RGB", (224, 224), color=(73, 109, 137))
        img.save(tmp_path)

    try:
        res = pipeline.predict(tmp_path)
        assert res["status"] == "success"
        assert "cnn" in res
        assert res["cnn"]["prediction"] in pipeline.classes
        assert 0.0 <= res["cnn"]["confidence"] <= 1.0
        assert len(res["cnn"]["probabilities"]) == 38

        # If VQC is initialized
        if res["vqc"] is not None:
            assert res["vqc"]["prediction"] in pipeline.classes
            assert 0.0 <= res["vqc"]["confidence"] <= 1.0
            assert len(res["vqc"]["pca_features"]) == 4
            assert len(res["vqc"]["basis_probabilities"]) == 16
            assert len(res["vqc"]["probabilities"]) == 38

        # If Hybrid is initialized
        if res["hybrid"] is not None:
            assert res["hybrid"]["prediction"] in pipeline.classes
            assert 0.0 <= res["hybrid"]["confidence"] <= 1.0
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
