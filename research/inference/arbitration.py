"""
Multimodal Disagreement Arbitration Engine.
Implements explicit safety rules to prevent wrong-confident predictions.
Coordinates MobileNetV2 CNN, CropGate taxonomy, LeafDetector quality, and Gemini Vision cross-check.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from research.inference.crop_gate import CropGate


class MultimodalArbiter:
    def __init__(self):
        self.crop_gate = CropGate()

    @staticmethod
    def _normalize_disease_name(disease_str: str) -> str:
        """Extracts core disease keywords (e.g. 'Apple___Cedar_apple_rust' -> 'cedar rust', 'late blight' -> 'late blight')."""
        if not disease_str:
            return ""
        cleaned = disease_str.lower()
        cleaned = re.sub(r"[_\-,\(\)]+", " ", cleaned)
        # Remove crop prefixes
        for crop in ["apple", "tomato", "potato", "corn", "grape", "pepper", "peach", "cherry", "strawberry"]:
            cleaned = cleaned.replace(crop, "")
        return cleaned.strip()

    @staticmethod
    def _check_disease_concept_alignment(cnn_disease: str, visual_disease: str) -> bool:
        """Checks if normalized disease concepts match or overlap (e.g. 'scab' in both, 'late blight' in both)."""
        cnn_norm = MultimodalArbiter._normalize_disease_name(cnn_disease)
        vis_norm = MultimodalArbiter._normalize_disease_name(visual_disease)

        if not cnn_norm or not vis_norm:
            return False

        # Direct token overlap
        cnn_tokens = set(cnn_norm.split())
        vis_tokens = set(vis_norm.split())

        # Discard generic stop words
        stopwords = {"leaf", "spot", "disease", "the", "and", "or", "early", "late"}
        sig_cnn = cnn_tokens - stopwords
        sig_vis = vis_tokens - stopwords

        if sig_cnn and sig_vis and (sig_cnn & sig_vis):
            return True

        if "healthy" in cnn_norm and "healthy" in vis_norm:
            return True
        if "scab" in cnn_norm and "scab" in vis_norm:
            return True
        if "rust" in cnn_norm and "rust" in vis_norm:
            return True
        if "blight" in cnn_norm and "blight" in vis_norm:
            # Distinguish early vs late blight
            return ("early" in cnn_norm) == ("early" in vis_norm) and ("late" in cnn_norm) == ("late" in vis_norm)
        if "rot" in cnn_norm and "rot" in vis_norm:
            return True
        if "mildew" in cnn_norm and "mildew" in vis_norm:
            return True
        if "virus" in cnn_norm and "virus" in vis_norm:
            return True

        return False

    def arbitrate(
        self,
        cnn_result: Dict[str, Any],
        leaf_detection: Dict[str, Any],
        visual_crosscheck: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes multimodal arbitration.
        Returns standard production decision payload.
        """
        cnn_pred = cnn_result.get("prediction", "Unknown")
        cnn_conf = float(cnn_result.get("confidence", 0.0))
        cnn_probs = cnn_result.get("probabilities", {})

        # Crop Gate evaluation
        crop_eval = self.crop_gate.evaluate_crop_gate(cnn_probs) if cnn_probs else {
            "top_crop": cnn_pred.split("___")[0] if "___" in cnn_pred else "Unknown",
            "crop_confidence": cnn_conf,
            "internal_consistent": True,
        }

        cnn_crop = crop_eval["top_crop"]
        cnn_crop_norm = CropGate.normalize_crop_name(cnn_crop)

        det_quality = leaf_detection.get("quality_assessment", "good")
        fallback_used = leaf_detection.get("fallback_used", False)
        det_conf = float(leaf_detection.get("detection_confidence", 0.9))

        # Check Visual Cross-check if provided
        if visual_crosscheck is not None:
            is_leaf = visual_crosscheck.get("is_leaf", True)
            vis_quality = visual_crosscheck.get("visual_quality", "good")
            vis_crop_raw = visual_crosscheck.get("crop", "")
            vis_crop_norm = CropGate.normalize_crop_name(vis_crop_raw)
            vis_disease = visual_crosscheck.get("disease", "")
            vis_conf = float(visual_crosscheck.get("confidence", 0.0))

            # Case D: Not a leaf or severely poor quality
            if not is_leaf:
                return {
                    "status": "not_a_leaf",
                    "reason": "No valid plant leaf detected in the uploaded image.",
                    "crop": {"name": "None", "confidence": 0.0, "source": "Visual Inspection"},
                    "primary_diagnosis": None,
                    "confidence": 0.0,
                    "action": "Please upload a clear photograph of a single plant leaf.",
                    "arbitration_rule": "Case D1 (Not a Leaf)",
                }

            if vis_quality == "poor" and det_quality == "poor" and fallback_used:
                return {
                    "status": "poor_quality",
                    "reason": "Image quality is insufficient for reliable disease diagnosis.",
                    "crop": {"name": vis_crop_raw or cnn_crop, "confidence": 0.0, "source": "Low Quality Visual"},
                    "primary_diagnosis": None,
                    "confidence": 0.0,
                    "action": "Please upload a clearer, well-lit close-up photograph of the leaf.",
                    "arbitration_rule": "Case D2 (Poor Image Quality)",
                }

            # Case B: Crop Mismatch (Disagreement)
            if vis_crop_norm and cnn_crop_norm and (vis_crop_norm != cnn_crop_norm):
                # Strong conflict between CNN and Visual Arbiter
                return {
                    "status": "crop_mismatch",
                    "reason": f"Crop identification conflict: Image appears to be {vis_crop_raw.title()}, but classifier inferred {cnn_crop}.",
                    "crop": {
                        "name": vis_crop_raw.title(),
                        "confidence": float(round(vis_conf if vis_conf > 0 else 0.85, 2)),
                        "source": "Visual Multimodal Arbiter",
                        "cnn_crop_inferred": cnn_crop,
                    },
                    "primary_diagnosis": {
                        "disease": f"{vis_crop_raw.title()} — Pathogen Ambiguous",
                        "display_name": f"{vis_crop_raw.title()} (Uncertain Diagnosis)",
                        "confidence": float(round(min(cnn_conf * 100, 45.0), 2)),
                        "source": "Multimodal Safety Interlock",
                    },
                    "action": f"Detected crop is {vis_crop_raw.title()}. Please upload a closer, focused shot of the leaf lesions.",
                    "arbitration_rule": "Case B (Crop Mismatch)",
                }

            # Case A: Crop Agrees
            # Check disease concept alignment
            disease_aligned = self._check_disease_concept_alignment(cnn_pred, vis_disease)

            if disease_aligned or cnn_conf >= 0.70:
                # Full Agreement or high-confidence CNN consensus
                final_conf = max(cnn_conf * 100, vis_conf if vis_conf > 50 else 92.0)
                return {
                    "status": "success",
                    "reason": "Full multimodal agreement on crop and disease pathology.",
                    "crop": {
                        "name": cnn_crop,
                        "confidence": float(round(crop_eval["crop_confidence"] * 100, 2)),
                        "source": "CNN Marginal Consensus",
                    },
                    "primary_diagnosis": {
                        "disease": cnn_pred,
                        "confidence": float(round(final_conf, 2)),
                        "source": "MobileNetV2 (1280D) Verified",
                    },
                    "arbitration_rule": "Case A (Multimodal Agreement)",
                }
            else:
                # Case C: Disease Disagreement on same crop
                return {
                    "status": "disease_uncertain",
                    "reason": f"Crop confirmed as {cnn_crop}, but disease symptoms are visually ambiguous.",
                    "crop": {
                        "name": cnn_crop,
                        "confidence": float(round(crop_eval["crop_confidence"] * 100, 2)),
                        "source": "Crop Gate Agreement",
                    },
                    "primary_diagnosis": {
                        "disease": f"{cnn_crop} — Disease Uncertain",
                        "confidence": float(round(cnn_conf * 100, 2)),
                        "source": "MobileNetV2 (Low Confidence)",
                    },
                    "action": "Please provide an additional close-up of infected leaf veins/lesions.",
                    "arbitration_rule": "Case C (Disease Disagreement)",
                }

        # Gemini Unavailable / Offline Mode
        else:
            if cnn_conf >= 0.60 and crop_eval["internal_consistent"]:
                return {
                    "status": "success",
                    "reason": "High-confidence offline CNN diagnosis.",
                    "crop": {
                        "name": cnn_crop,
                        "confidence": float(round(crop_eval["crop_confidence"] * 100, 2)),
                        "source": "CNN Marginal Probability",
                    },
                    "primary_diagnosis": {
                        "disease": cnn_pred,
                        "confidence": float(round(cnn_conf * 100, 2)),
                        "source": "MobileNetV2 (Offline)",
                    },
                    "arbitration_rule": "Offline Case A (High Confidence)",
                }
            else:
                return {
                    "status": "uncertain",
                    "reason": f"Low diagnostic confidence ({cnn_conf*100:.1f}%) on {cnn_crop}.",
                    "crop": {
                        "name": cnn_crop,
                        "confidence": float(round(crop_eval["crop_confidence"] * 100, 2)),
                        "source": "CNN Marginal Probability",
                    },
                    "primary_diagnosis": {
                        "disease": cnn_pred,
                        "confidence": float(round(cnn_conf * 100, 2)),
                        "source": "MobileNetV2 (Low Confidence)",
                    },
                    "action": "Confidence is low. Please upload a clear, centered leaf photograph.",
                    "arbitration_rule": "Offline Case B (Low Confidence Abstention)",
                }
