import pytest
from research.inference.real_world_diagnostic import RealWorldDiagnostic
from research.inference.predict import PredictionPipeline

def test_prediction_pipeline_returns_leaf_detection_and_crop():
    pipeline = PredictionPipeline()
    res = pipeline.predict("research/test_sample_leaf.png")
    
    assert "leaf_detection" in res
    assert "crop" in res
    assert "bbox" in res["leaf_detection"]
    assert "detection_confidence" in res["leaf_detection"]
    assert "name" in res["crop"]
    assert "confidence" in res["crop"]
    assert res["cnn"]["confidence"] > 0

def test_real_world_diagnostic_execution():
    diag = RealWorldDiagnostic()
    out = diag.diagnose_image("research/test_sample_leaf.png")
    
    assert "raw_prediction" in out
    assert "cropped_prediction" in out
    assert "preprocessing_changed_prediction" in out
    assert "leaf_detection" in out
    assert out["leaf_detection"]["quality"] in ["good", "moderate", "poor"]
