import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import torch
from PIL import Image
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import Statevector
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]


class PredictionPipeline:
    def __init__(self):
        self.checkpoint_path = ROOT / "research/models/mobilenetv2_best.pt"
        if not self.checkpoint_path.exists():
            self.checkpoint_path = ROOT / "research/mobilenetv2_best.pt"

        self.scaler_path = ROOT / "research/models/feature_scaler.joblib"
        self.pca_path = ROOT / "research/models/feature_pca.joblib"
        self.vqc_params_path = ROOT / "research/models/vqc_params.json"
        self.hybrid_fusion_path = ROOT / "research/models/hybrid_fusion.pt"

        self._load_models()

    def _load_models(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.classes = ckpt.get("label_names") or ckpt.get("classes")
        self.num_classes = len(self.classes)

        # 1. Classical CNN
        self.cnn = models.mobilenet_v2(weights=None)
        in_feat = self.cnn.classifier[1].in_features
        self.cnn.classifier[1] = nn.Linear(in_feat, self.num_classes)
        self.cnn.load_state_dict(ckpt.get("model_state_dict") or ckpt.get("state_dict"))
        self.cnn.eval()

        # Separate feature extractor
        self.feature_extractor = models.mobilenet_v2(weights=None)
        self.feature_extractor.classifier[1] = nn.Linear(in_feat, self.num_classes)
        self.feature_extractor.load_state_dict(ckpt.get("model_state_dict") or ckpt.get("state_dict"))
        self.feature_extractor.classifier = nn.Identity()
        self.feature_extractor.eval()

        self.classifier_head = self.cnn.classifier

        # 2. Scaler and PCA
        self.scaler = joblib.load(self.scaler_path) if self.scaler_path.exists() else None
        self.pca = joblib.load(self.pca_path) if self.pca_path.exists() else None

        # 3. Quantum VQC
        self.vqc_params = None
        self.projection_head = None
        if self.vqc_params_path.exists():
            with open(self.vqc_params_path, "r", encoding="utf-8") as f:
                self.vqc_params = json.load(f)

            self.theta = np.array(self.vqc_params["theta"])
            self.vqc_fmap = ZZFeatureMap(4, reps=1, entanglement="full")
            self.vqc_ansatz = RealAmplitudes(4, reps=1, entanglement="full")
            self.vqc_base = QuantumCircuit(4)
            self.vqc_base.compose(self.vqc_fmap, inplace=True)
            self.vqc_base.compose(self.vqc_ansatz, inplace=True)

            self.projection_head = nn.Linear(16, self.num_classes)
            proj_state = {
                "weight": torch.tensor(self.vqc_params["projection_weight"], dtype=torch.float32),
                "bias": torch.tensor(self.vqc_params["projection_bias"], dtype=torch.float32),
            }
            self.projection_head.load_state_dict(proj_state)
            self.projection_head.eval()

        # 4. Learned Hybrid Fusion
        self.fusion_model = None
        if self.hybrid_fusion_path.exists():
            state = torch.load(self.hybrid_fusion_path, map_location="cpu", weights_only=True)
            # Normalize keys if needed
            clean_state = {k.replace("fusion.", ""): v for k, v in state.items()}
            self.fusion_model = nn.Sequential(
                nn.Linear(self.num_classes + 16, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_classes),
            )
            self.fusion_model.load_state_dict(clean_state)
            self.fusion_model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Preloaded Real-World Robustness Helpers
        from research.inference.crop_gate import CropGate
        from research.inference.leaf_detector import LeafDetector
        self.leaf_detector = LeafDetector()
        self.crop_gate = CropGate(self.checkpoint_path)

    def predict(self, image_path: str | Path) -> Dict[str, Any]:
        with Image.open(image_path).convert("RGB") as img:
            # 1. Leaf Detection & Isolation
            detection = self.leaf_detector.detect_and_isolate_leaf(img)
            
            # Select input image: use isolated leaf if detection quality is good, else raw
            target_img = detection["cropped_image"] if (detection["quality_assessment"] == "good" and not detection["fallback_used"]) else img
            x_tensor = self.transform(target_img).unsqueeze(0)

        with torch.no_grad():
            feat_1280 = self.feature_extractor(x_tensor)
            cnn_logits = self.classifier_head(feat_1280)
            cnn_probs = torch.softmax(cnn_logits, dim=-1)[0].numpy()

        cnn_idx = int(np.argmax(cnn_probs))
        cnn_pred = self.classes[cnn_idx]
        cnn_conf = float(cnn_probs[cnn_idx])

        # Evaluate Crop Gate
        crop_eval = self.crop_gate.evaluate_crop_gate(cnn_probs)

        result: Dict[str, Any] = {
            "status": "success",
            "model": "Hybrid Quantum-Classical Architecture",
            "leaf_detection": {
                "bbox": detection["bbox"],
                "normalized_bbox": detection["normalized_bbox"],
                "detection_confidence": detection["detection_confidence"],
                "fallback_used": detection["fallback_used"],
                "quality": detection["quality_assessment"],
                "leaf_area_ratio": detection["leaf_area_ratio"],
            },
            "crop": {
                "name": crop_eval["top_crop"],
                "confidence": crop_eval["crop_confidence"],
                "second_crop": crop_eval["second_crop"],
                "crop_margin": crop_eval["crop_margin"],
                "internal_consistent": crop_eval["internal_consistent"],
            },
            "cnn": {
                "model": "MobileNetV2 (1280D)",
                "prediction": cnn_pred,
                "confidence": cnn_conf,
                "probabilities": {c: float(p) for c, p in zip(self.classes, cnn_probs)},
            },
            "vqc": None,
            "hybrid": None,
        }

        # If Scaler, PCA, and VQC are ready
        if self.scaler is not None and self.pca is not None and self.projection_head is not None:
            feat_np = feat_1280.numpy()
            scaled_feat = self.scaler.transform(feat_np)
            pca_4d = self.pca.transform(scaled_feat)[0]

            # Compute 4-qubit statevector probabilities
            params = list(pca_4d[:4]) + list(self.theta)
            qc = self.vqc_base.assign_parameters(params)
            q_probs = Statevector.from_instruction(qc).probabilities()

            # Trainable classical projection
            with torch.no_grad():
                q_probs_t = torch.tensor(q_probs, dtype=torch.float32).unsqueeze(0)
                vqc_logits = self.projection_head(q_probs_t)
                vqc_probs = torch.softmax(vqc_logits, dim=-1)[0].numpy()

            vqc_idx = int(np.argmax(vqc_probs))
            vqc_pred = self.classes[vqc_idx]
            vqc_conf = float(vqc_probs[vqc_idx])

            result["vqc"] = {
                "backend": "Qiskit Statevector Simulator (4 Qubits)",
                "num_qubits": 4,
                "feature_map": "ZZFeatureMap (reps=1)",
                "ansatz": "RealAmplitudes (reps=1)",
                "pca_features": [round(float(v), 4) for v in pca_4d],
                "basis_probabilities": [round(float(p), 4) for p in q_probs],
                "prediction": vqc_pred,
                "confidence": vqc_conf,
                "probabilities": {c: float(p) for c, p in zip(self.classes, vqc_probs)},
            }

            # If Learned Fusion model is ready
            if self.fusion_model is not None:
                with torch.no_grad():
                    combined_input = torch.cat([cnn_logits, q_probs_t], dim=-1)
                    fused_logits = self.fusion_model(combined_input)
                    fused_probs = torch.softmax(fused_logits, dim=-1)[0].numpy()

                fused_idx = int(np.argmax(fused_probs))
                fused_pred = self.classes[fused_idx]
                fused_conf = float(fused_probs[fused_idx])

                result["hybrid"] = {
                    "fusion_model": "Learned Hybrid Representation Fusion",
                    "prediction": fused_pred,
                    "confidence": fused_conf,
                    "probabilities": {c: float(p) for c, p in zip(self.classes, fused_probs)},
                }
            else:
                result["hybrid"] = {
                    "fusion_model": "Confidence Fallback",
                    "prediction": cnn_pred,
                    "confidence": cnn_conf,
                }

        return result


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python research/inference/predict.py <image_path>")

    image_path = Path(sys.argv[1])
    pipeline = PredictionPipeline()
    res = pipeline.predict(image_path)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
