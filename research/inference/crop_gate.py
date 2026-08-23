"""
Deterministic Crop Identification Gate.
Derived directly from the 38-class PlantVillage taxonomy in mobilenetv2_best.pt.
Calculates crop-level marginal probabilities and detects crop-level disagreements.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


class CropGate:
    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path or (ROOT / "research/models/mobilenetv2_best.pt")
        if not self.checkpoint_path.exists():
            self.checkpoint_path = ROOT / "research/mobilenetv2_best.pt"

        self.classes, self.crop_to_classes, self.class_to_crop = self._build_taxonomy()
        self.supported_crops = sorted(list(self.crop_to_classes.keys()))

    def _build_taxonomy(self) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            classes = ckpt.get("label_names") or ckpt.get("classes") or []
        else:
            # Fallback to split_manifest.json
            manifest_path = ROOT / "research/split_manifest.json"
            with open(manifest_path, "r", encoding="utf-8") as f:
                classes = json.load(f)["classes"]

        if not classes or len(classes) != 38:
            raise ValueError(f"Expected 38 classes, found {len(classes)} in {self.checkpoint_path}")

        crop_to_classes: Dict[str, List[str]] = {}
        class_to_crop: Dict[str, str] = {}

        for cls_name in classes:
            # Format is typically Crop___Disease
            if "___" in cls_name:
                crop = cls_name.split("___")[0]
            else:
                crop = cls_name.split("_")[0]

            class_to_crop[cls_name] = crop
            if crop not in crop_to_classes:
                crop_to_classes[crop] = []
            crop_to_classes[crop].append(cls_name)

        return classes, crop_to_classes, class_to_crop

    def calculate_marginal_crop_probabilities(
        self, class_probabilities: Dict[str, float] | np.ndarray | torch.Tensor
    ) -> Dict[str, float]:
        """Calculates marginal P(Crop) = Sum_{c in Crop} P(Class_c)."""
        if isinstance(class_probabilities, (np.ndarray, list)):
            class_probs = {c: float(p) for c, p in zip(self.classes, class_probabilities)}
        elif isinstance(class_probabilities, torch.Tensor):
            probs_np = class_probabilities.detach().cpu().numpy().flatten()
            class_probs = {c: float(p) for c, p in zip(self.classes, probs_np)}
        else:
            class_probs = class_probabilities

        crop_marginals: Dict[str, float] = {crop: 0.0 for crop in self.supported_crops}
        for cls_name, prob in class_probs.items():
            crop = self.class_to_crop.get(cls_name)
            if crop in crop_marginals:
                crop_marginals[crop] += float(prob)

        # Normalize sum to 1.0
        total = sum(crop_marginals.values())
        if total > 0:
            crop_marginals = {k: v / total for k, v in crop_marginals.items()}

        return crop_marginals

    def evaluate_crop_gate(
        self, class_probabilities: Dict[str, float] | np.ndarray | torch.Tensor
    ) -> Dict[str, Any]:
        """Evaluates crop identity, marginal confidence, and checks internal consistency."""
        crop_marginals = self.calculate_marginal_crop_probabilities(class_probabilities)
        
        # Sort crops by marginal probability
        sorted_crops = sorted(crop_marginals.items(), key=lambda x: x[1], reverse=True)
        top_crop, top_crop_conf = sorted_crops[0]
        second_crop, second_crop_conf = sorted_crops[1] if len(sorted_crops) > 1 else ("None", 0.0)

        # Get top disease class
        if isinstance(class_probabilities, dict):
            top_class = max(class_probabilities.items(), key=lambda x: x[1])
            top_disease, top_disease_conf = top_class[0], float(top_class[1])
        else:
            probs_np = np.array(class_probabilities).flatten()
            top_idx = int(np.argmax(probs_np))
            top_disease, top_disease_conf = self.classes[top_idx], float(probs_np[top_idx])

        disease_implied_crop = self.class_to_crop.get(top_disease, "Unknown")
        internal_crop_consistent = (disease_implied_crop == top_crop)

        # Entropy / Margin as crop ambiguity indicator
        crop_margin = top_crop_conf - second_crop_conf

        return {
            "top_crop": top_crop,
            "crop_confidence": float(top_crop_conf),
            "second_crop": second_crop,
            "second_crop_confidence": float(second_crop_conf),
            "crop_margin": float(crop_margin),
            "disease_implied_crop": disease_implied_crop,
            "top_disease": top_disease,
            "disease_confidence": float(top_disease_conf),
            "internal_consistent": internal_crop_consistent,
            "marginal_probabilities": crop_marginals,
        }

    @staticmethod
    def normalize_crop_name(name: str) -> str:
        """Normalizes various crop naming formats (e.g. 'Corn_(maize)' -> 'corn', 'Pepper,_bell' -> 'pepper')."""
        if not name:
            return ""
        cleaned = name.lower()
        cleaned = re.sub(r"[_\-,\(\)]+", " ", cleaned).strip()
        # Common aliases
        if "corn" in cleaned or "maize" in cleaned:
            return "corn"
        if "pepper" in cleaned or "bell" in cleaned:
            return "pepper"
        if "apple" in cleaned:
            return "apple"
        if "potato" in cleaned:
            return "potato"
        if "tomato" in cleaned:
            return "tomato"
        if "grape" in cleaned:
            return "grape"
        if "cherry" in cleaned:
            return "cherry"
        if "peach" in cleaned:
            return "peach"
        if "strawberry" in cleaned:
            return "strawberry"
        if "orange" in cleaned or "citrus" in cleaned:
            return "orange"
        if "blueberry" in cleaned:
            return "blueberry"
        if "soybean" in cleaned or "soy" in cleaned:
            return "soybean"
        if "squash" in cleaned:
            return "squash"
        if "raspberry" in cleaned:
            return "raspberry"
        return cleaned.split()[0] if cleaned.split() else cleaned
