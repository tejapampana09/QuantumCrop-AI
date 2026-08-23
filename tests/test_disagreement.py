import pytest
from research.inference.arbitration import MultimodalArbiter

def test_case_a_full_agreement():
    arbiter = MultimodalArbiter()
    cnn_res = {
        "prediction": "Apple___Apple_scab",
        "confidence": 0.94,
        "probabilities": {"Apple___Apple_scab": 0.94, "Apple___Black_rot": 0.06}
    }
    leaf_det = {"quality_assessment": "good", "detection_confidence": 0.95, "fallback_used": False}
    vis_cross = {
        "is_leaf": True,
        "crop": "Apple",
        "disease": "Apple Scab",
        "visual_quality": "good",
        "confidence": 95.0
    }
    
    out = arbiter.arbitrate(cnn_res, leaf_det, vis_cross)
    assert out["status"] == "success"
    assert "Apple" in out["crop"]["name"]

def test_case_b_crop_mismatch():
    arbiter = MultimodalArbiter()
    # CNN predicted Potato, but visual cross-check identified Apple
    cnn_res = {
        "prediction": "Potato___Early_blight",
        "confidence": 0.35,
        "probabilities": {"Potato___Early_blight": 0.35, "Tomato___Early_blight": 0.20}
    }
    leaf_det = {"quality_assessment": "good", "detection_confidence": 0.90, "fallback_used": False}
    vis_cross = {
        "is_leaf": True,
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "visual_quality": "good",
        "confidence": 92.0
    }
    
    out = arbiter.arbitrate(cnn_res, leaf_det, vis_cross)
    assert out["status"] == "crop_mismatch"
    assert "Apple" in out["crop"]["name"]
    assert "conflict" in out["reason"].lower()

def test_case_d_not_a_leaf():
    arbiter = MultimodalArbiter()
    cnn_res = {
        "prediction": "Tomato___healthy",
        "confidence": 0.30,
        "probabilities": {"Tomato___healthy": 0.30}
    }
    leaf_det = {"quality_assessment": "poor", "detection_confidence": 0.20, "fallback_used": True}
    vis_cross = {
        "is_leaf": False,
        "crop": "None",
        "disease": "None",
        "visual_quality": "poor",
        "confidence": 0.0
    }
    
    out = arbiter.arbitrate(cnn_res, leaf_det, vis_cross)
    assert out["status"] == "not_a_leaf"
    assert out["primary_diagnosis"] is None

def test_offline_low_confidence_abstention():
    arbiter = MultimodalArbiter()
    cnn_res = {
        "prediction": "Corn_(maize)___Common_rust_",
        "confidence": 0.32,
        "probabilities": {"Corn_(maize)___Common_rust_": 0.32}
    }
    leaf_det = {"quality_assessment": "good", "detection_confidence": 0.80, "fallback_used": False}
    
    # Offline: no visual cross-check provided
    out = arbiter.arbitrate(cnn_res, leaf_det, visual_crosscheck=None)
    assert out["status"] == "uncertain"
