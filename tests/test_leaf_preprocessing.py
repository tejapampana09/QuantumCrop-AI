import pytest
import numpy as np
from PIL import Image
from research.inference.leaf_detector import LeafDetector

def test_leaf_detector_on_synthetic_leaf():
    detector = LeafDetector()
    # Create image with dark green centered circle on white background
    img_np = np.ones((200, 200, 3), dtype=np.uint8) * 240
    y, x = np.ogrid[:200, :200]
    mask = (x - 100)**2 + (y - 100)**2 <= 50**2
    img_np[mask] = [34, 139, 34] # Forest green
    
    img_pil = Image.fromarray(img_np)
    res = detector.detect_and_isolate_leaf(img_pil)
    
    assert res["quality_assessment"] in ["good", "moderate"]
    assert res["fallback_used"] is False
    assert len(res["bbox"]) == 4
    assert res["leaf_area_ratio"] > 0.10

def test_leaf_detector_fallback_on_blank():
    detector = LeafDetector()
    # Pure white image
    blank_np = np.ones((150, 150, 3), dtype=np.uint8) * 255
    res = detector.detect_and_isolate_leaf(blank_np)
    
    assert res["fallback_used"] is True
    assert res["quality_assessment"] == "poor"
    assert res["detection_confidence"] < 0.40

def test_leaf_detector_on_real_sample():
    detector = LeafDetector()
    img_path = "research/test_sample_leaf.png"
    img = Image.open(img_path)
    res = detector.detect_and_isolate_leaf(img)
    
    assert res["quality_assessment"] == "good"
    assert res["detection_confidence"] > 0.80
    assert res["cropped_image"].size[0] > 0
