import pytest
import numpy as np
import torch
from research.inference.crop_gate import CropGate

def test_crop_taxonomy_initialization():
    gate = CropGate()
    assert len(gate.classes) == 38
    assert len(gate.supported_crops) == 14
    assert "Apple" in gate.supported_crops
    assert "Tomato" in gate.supported_crops
    assert "Potato" in gate.supported_crops
    assert "Corn_(maize)" in gate.supported_crops
    assert "Grape" in gate.supported_crops
    assert "Pepper,_bell" in gate.supported_crops

def test_crop_marginal_calculation():
    gate = CropGate()
    # Mock softmax distribution where Tomato classes dominate
    probs = {c: 0.001 for c in gate.classes}
    probs["Tomato___Late_blight"] = 0.60
    probs["Tomato___Early_blight"] = 0.20
    probs["Tomato___healthy"] = 0.10
    
    marginals = gate.calculate_marginal_crop_probabilities(probs)
    assert marginals["Tomato"] > 0.85
    assert marginals["Apple"] < 0.05

def test_crop_gate_consistency_evaluation():
    gate = CropGate()
    # Consistent sample
    probs = {c: 0.01 for c in gate.classes}
    probs["Apple___Apple_scab"] = 0.60
    
    eval_res = gate.evaluate_crop_gate(probs)
    assert eval_res["top_crop"] == "Apple"
    assert eval_res["top_disease"] == "Apple___Apple_scab"
    assert eval_res["internal_consistent"] is True

def test_crop_name_normalization():
    assert CropGate.normalize_crop_name("Corn_(maize)") == "corn"
    assert CropGate.normalize_crop_name("Pepper,_bell") == "pepper"
    assert CropGate.normalize_crop_name("Apple___Apple_scab") == "apple"
    assert CropGate.normalize_crop_name("Tomato") == "tomato"
