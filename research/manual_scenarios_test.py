import sys
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.inference.predict import PredictionPipeline
from research.inference.arbitration import MultimodalArbiter
from research.inference.leaf_detector import LeafDetector

pipeline = PredictionPipeline()
arbiter = MultimodalArbiter()
detector = LeafDetector()

print("=== 1. TEST: APPLE OUTDOOR LEAF ===")
apple_outdoor_path = r"C:\Users\tejap\.gemini\antigravity\brain\0dce3af8-ab4d-4edb-955b-4fba2d3bb736\.user_uploaded\media_1787511775311.png"
res1 = pipeline.predict(apple_outdoor_path)
dec1 = arbiter.arbitrate(
    cnn_result=res1["cnn"],
    leaf_detection=res1["leaf_detection"],
    visual_crosscheck={"is_leaf": True, "crop": "Apple", "disease": "Cedar Apple Rust", "visual_quality": "good", "confidence": 95.0}
)
print("Apple Outdoor -> Status:", dec1["status"])
print("Reason:", dec1["reason"])

print("\n=== 2. TEST: NON-LEAF (BLANK/OBJECT) ===")
blank_img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8)*255)
blank_path = ROOT / "research/eval_samples/blank_non_leaf.png"
blank_img.save(blank_path)
res2 = pipeline.predict(blank_path)
dec2 = arbiter.arbitrate(
    cnn_result=res2["cnn"],
    leaf_detection=res2["leaf_detection"],
    visual_crosscheck={"is_leaf": False, "crop": "None", "disease": "None", "visual_quality": "poor", "confidence": 0.0}
)
print("Non-leaf -> Status:", dec2["status"])

print("\n=== 3. TEST: BLURRY LEAF ===")
blur_img = Image.open(apple_outdoor_path).filter(ImageFilter.GaussianBlur(radius=15))
blur_path = ROOT / "research/eval_samples/blurry_leaf.png"
blur_img.save(blur_path)
res3 = pipeline.predict(blur_path)
dec3 = arbiter.arbitrate(
    cnn_result=res3["cnn"],
    leaf_detection=res3["leaf_detection"],
    visual_crosscheck={"is_leaf": True, "crop": "Apple", "disease": "Unknown", "visual_quality": "poor", "confidence": 20.0}
)
print("Blurry Leaf -> Status:", dec3["status"])

print("\n=== 4. TEST: LAB BENCHMARK APPLE SCAB ===")
scab_path = ROOT / "research/sample_apple_scab.png"
res4 = pipeline.predict(scab_path)
dec4 = arbiter.arbitrate(
    cnn_result=res4["cnn"],
    leaf_detection=res4["leaf_detection"],
    visual_crosscheck={"is_leaf": True, "crop": "Apple", "disease": "Apple Scab", "visual_quality": "good", "confidence": 95.0}
)
print("Lab Apple Scab -> Status:", dec4["status"])
print("Disease:", dec4["primary_diagnosis"]["disease"], dec4["primary_diagnosis"]["confidence"])
