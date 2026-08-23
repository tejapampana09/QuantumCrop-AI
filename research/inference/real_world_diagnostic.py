"""
Real-World Diagnostic Pipeline.
Compares Raw Image vs Preprocessed/Cropped Image through MobileNetV2 and CropGate.
Analyzes whether leaf isolation changes prediction or improves crop consistency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.inference.crop_gate import CropGate
from research.inference.leaf_detector import LeafDetector
from research.inference.predict import PredictionPipeline


class RealWorldDiagnostic:
    def __init__(self):
        self.pipeline = PredictionPipeline()
        self.crop_gate = CropGate()
        self.leaf_detector = LeafDetector()

    def diagnose_image(self, image_path: str | Path) -> Dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        raw_img = Image.open(image_path).convert("RGB")

        # 1. Raw Image Inference
        raw_result = self.pipeline.predict(image_path)
        raw_pred = raw_result["cnn"]["prediction"]
        raw_conf = raw_result["cnn"]["confidence"]
        raw_probs = raw_result["cnn"]["probabilities"]
        raw_crop_eval = self.crop_gate.evaluate_crop_gate(raw_probs)

        # 2. Leaf Detection & Isolation
        detection_res = self.leaf_detector.detect_and_isolate_leaf(raw_img)
        cropped_pil = detection_res["cropped_image"]

        # 3. Cropped Image Inference
        # Save temporary cropped image for predict pipeline or pass via tensor
        temp_cropped_path = image_path.parent / f"_diag_cropped_{image_path.name}"
        try:
            cropped_pil.save(temp_cropped_path)
            cropped_result = self.pipeline.predict(temp_cropped_path)
        finally:
            if temp_cropped_path.exists():
                temp_cropped_path.unlink()

        cropped_pred = cropped_result["cnn"]["prediction"]
        cropped_conf = cropped_result["cnn"]["confidence"]
        cropped_probs = cropped_result["cnn"]["probabilities"]
        cropped_crop_eval = self.crop_gate.evaluate_crop_gate(cropped_probs)

        changed = (raw_pred != cropped_pred)

        return {
            "image_path": str(image_path),
            "raw_prediction": raw_pred,
            "raw_confidence": float(round(raw_conf, 4)),
            "raw_crop": raw_crop_eval["top_crop"],
            "raw_crop_confidence": float(round(raw_crop_eval["crop_confidence"], 4)),
            "raw_crop_consistent": raw_crop_eval["internal_consistent"],
            "cropped_prediction": cropped_pred,
            "cropped_confidence": float(round(cropped_conf, 4)),
            "cropped_crop": cropped_crop_eval["top_crop"],
            "cropped_crop_confidence": float(round(cropped_crop_eval["crop_confidence"], 4)),
            "cropped_crop_consistent": cropped_crop_eval["internal_consistent"],
            "preprocessing_changed_prediction": changed,
            "leaf_detection": {
                "bbox": detection_res["bbox"],
                "normalized_bbox": detection_res["normalized_bbox"],
                "detection_confidence": detection_res["detection_confidence"],
                "fallback_used": detection_res["fallback_used"],
                "quality": detection_res["quality_assessment"],
                "leaf_area_ratio": detection_res["leaf_area_ratio"],
            },
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python research/inference/real_world_diagnostic.py <image_path>")
        sys.exit(1)

    diagnostic = RealWorldDiagnostic()
    result = diagnostic.diagnose_image(sys.argv[1])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
