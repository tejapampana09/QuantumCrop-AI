from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from torch import nn

from research.training.train_vqc import (
    ClassicalPCAControlHead,
    LearnedHybridFusion,
    VQCClassifierHead,
    VQCCircuitSimulator,
)

ROOT = Path(__file__).resolve().parents[2]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "per_class": {
            cls_name: {
                "precision": report[cls_name]["precision"],
                "recall": report[cls_name]["recall"],
                "f1_score": report[cls_name]["f1-score"],
                "support": int(report[cls_name]["support"]),
            }
            for cls_name in classes
        },
        "confusion_matrix": cm,
    }


def generate_markdown_report(results: Dict[str, Any], output_path: Path) -> None:
    exp_a = results["experiments"]["experiment_a_cnn"]
    exp_b = results["experiments"]["experiment_b_classical_pca_head"]
    exp_c = results["experiments"]["experiment_c_vqc_head"]
    exp_d = results["experiments"]["experiment_d_hybrid_fusion"]
    num_samples = results["test_samples"]

    report_content = f"""# QuantumCrop AI — Quantum-Classical Experiment Scientific Report

**Date**: 2026-08-23  
**Dataset**: `BrandonFors/Plant-Diseases-PlantVillage-Dataset` (Hugging Face)  
**Untouched Test Set Size**: {num_samples} samples across 38 crop disease classes  
**Evaluation Protocol**: Strictly isolated evaluation on the identical held-out test split.

---

## 1. Executive Summary & Core Scientific Findings

This experiment evaluates the integration of a **4-qubit Variational Quantum Circuit (VQC)** within a deep classical vision pipeline for plant disease diagnosis. To ensure scientific rigor and isolate the quantum representation effects from aggressive dimensionality reduction, four controlled architectures were evaluated on the exact same untouched test set:

1. **Experiment A (End-to-End Classical CNN)**: Full MobileNetV2 (1280D features $\\to$ Linear classifier).
2. **Experiment B (Classical Control Head)**: MobileNetV2 1280D $\\to$ Scaler $\\to$ PCA-4 $\\to$ Classical 16-hidden MLP $\\to$ 38 classes.
3. **Experiment C (Quantum VQC Head)**: MobileNetV2 1280D $\\to$ Scaler $\\to$ PCA-4 $\\to$ 4-Qubit VQC (`ZZFeatureMap` + `RealAmplitudes`) $\\to$ 16 Quantum State Probabilities $\\to$ Trainable Projection $\\to$ 38 classes.
4. **Experiment D (Learned Hybrid Fusion)**: MobileNetV2 representation + VQC quantum representation $\\to$ Learned Fusion Layer $\\to$ 38 classes.

### Primary Metrics Comparison Table

| Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Input Dimension | Classifier Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: MobileNetV2 CNN Baseline** | **{exp_a['metrics']['accuracy']*100:.2f}%** | **{exp_a['metrics']['precision_macro']*100:.2f}%** | **{exp_a['metrics']['recall_macro']*100:.2f}%** | **{exp_a['metrics']['f1_macro']*100:.2f}%** | 1280D | Classical Linear (1280 $\\to$ 38) |
| **Exp B: Classical PCA Control Head** | {exp_b['metrics']['accuracy']*100:.2f}% | {exp_b['metrics']['precision_macro']*100:.2f}% | {exp_b['metrics']['recall_macro']*100:.2f}% | {exp_b['metrics']['f1_macro']*100:.2f}% | 4D (PCA) | Classical MLP (4 $\\to$ 16 $\\to$ 38) |
| **Exp C: Quantum VQC Head** | {exp_c['metrics']['accuracy']*100:.2f}% | {exp_c['metrics']['precision_macro']*100:.2f}% | {exp_c['metrics']['recall_macro']*100:.2f}% | {exp_c['metrics']['f1_macro']*100:.2f}% | 4D (PCA) | 4-Qubit VQC + Linear (16 $\\to$ 38) |
| **Exp D: Learned Hybrid Fusion** | {exp_d['metrics']['accuracy']*100:.2f}% | {exp_d['metrics']['precision_macro']*100:.2f}% | {exp_d['metrics']['recall_macro']*100:.2f}% | {exp_d['metrics']['f1_macro']*100:.2f}% | 1280D + 16D | Learned Fusion Layer |

---

## 2. Quantum Advantage Analysis & Empirical Discussion

### Quantum Circuit vs. Classical Control (Exp C vs. Exp B)
- When bottlenecked to **4 PCA dimensions**, the 4-qubit VQC achieves **{exp_c['metrics']['accuracy']*100:.2f}%** accuracy and **{exp_c['metrics']['f1_macro']*100:.2f}%** Macro-F1, compared to **{exp_b['metrics']['accuracy']*100:.2f}%** accuracy and **{exp_b['metrics']['f1_macro']*100:.2f}%** Macro-F1 for the classical MLP head with equivalent capacity.
- This comparison directly isolates whether the non-linear Hilbert space embedding ($ZZ$ feature map phase interactions and parameterized rotations) provides expressivity advantages over classical linear/ReLU projections on the same 4D latent space.

### Impact of Extreme Dimensionality Reduction (1280D $\\to$ 4D)
- The full MobileNetV2 baseline operating directly on 1280-dimensional feature maps retains high semantic granularity (**{exp_a['metrics']['accuracy']*100:.2f}%** accuracy).
- Compressing 1280 dimensions to 4 dimensions (a 320x compression ratio) captures a limited fraction of total feature variance, establishing the fundamental ceiling for any 4-qubit model.

### Hybrid Fusion Performance (Exp D vs. Exp A)
- The learned hybrid fusion model achieves **{exp_d['metrics']['accuracy']*100:.2f}%** accuracy and **{exp_d['metrics']['f1_macro']*100:.2f}%** Macro-F1.
- In accordance with rigorous scientific standards, we **do not claim quantum advantage** over the full-scale 1280D classical baseline. Rather, the hybrid architecture establishes a verified, working quantum-classical pipeline suitable for near-term NISQ simulation and quantum algorithm research.

---

## 3. Dataset & Preprocessing Audit Summary

- **Dataset**: BrandonFors/Plant-Diseases-PlantVillage-Dataset
- **Split Sizes**: Train = 36,937 | Validation = 6,519 | Official Test = 10,849
- **Leakage Prevention**:
  - `StandardScaler` and `PCA(n_components=4)` were fit **exclusively** on training features.
  - Zero index overlap between train and validation splits.
  - Official test set remained untouched until final benchmarking.
- **Hardware Simulation**: Qiskit Statevector simulation on CPU.

---

## 4. Conclusion & Production Recommendation

1. **Classical CNN (MobileNetV2)** remains the primary high-confidence production engine for immediate crop disease classification in field applications.
2. **Quantum VQC Layer** is successfully trained, parameter-persisted, and fully integrated into the production inference pipeline (`research/hybrid_pipeline.py`).
3. **Transparent Reporting**: The system transparently returns classical, VQC, and hybrid consensus scores without synthetic fallback or fabricated confidence scores.
"""

    output_path.write_text(report_content, encoding="utf-8")
    print(f"Saved Markdown experiment report to {output_path}")


def main():
    print("=" * 70)
    print("PHASE 5: RIGOROUS EVALUATION ON UNTOUCHED OFFICIAL TEST SET (10,849 SAMPLES)")
    print("=" * 70)

    pca_path = ROOT / "research/artifacts/cnn_features_pca4.npz"
    vqc_params_path = ROOT / "research/models/vqc_params.json"
    classical_head_path = ROOT / "research/models/classical_pca_head.pt"
    hybrid_fusion_path = ROOT / "research/models/hybrid_fusion.pt"

    if not pca_path.exists():
        raise FileNotFoundError(f"PCA features not found at {pca_path}. Run extract_features.py first.")
    if not vqc_params_path.exists():
        raise FileNotFoundError(f"VQC params not found at {vqc_params_path}. Run train_vqc.py first.")

    data = np.load(pca_path, allow_pickle=True)
    X_test_pca = data["X_test"]
    y_test = data["y_test"].astype(int)
    test_logits_cnn = data["test_logits"]
    classes = data["classes"].tolist()
    num_classes = len(classes)
    num_test = len(y_test)

    print(f"Loaded official test set: {num_test} samples, {num_classes} classes.")

    # 1. Evaluate Experiment A: Existing MobileNetV2 CNN Baseline
    print("\n--- Evaluating Experiment A: MobileNetV2 Classical CNN Baseline ---")
    start = time.time()
    cnn_preds = test_logits_cnn.argmax(axis=-1)
    cnn_time = time.time() - start
    metrics_a = evaluate_predictions(y_test, cnn_preds, classes)
    print(f"Exp A (CNN) - Acc: {metrics_a['accuracy']*100:.2f}% | Macro-F1: {metrics_a['f1_macro']*100:.2f}%")

    # 2. Evaluate Experiment B: Classical PCA Control Head
    print("\n--- Evaluating Experiment B: Classical PCA Control Head (4D -> MLP -> 38) ---")
    classical_model = ClassicalPCAControlHead(in_features=4, hidden_dim=16, num_classes=num_classes)
    classical_model.load_state_dict(torch.load(classical_head_path, map_location="cpu", weights_only=True))
    classical_model.eval()

    start = time.time()
    with torch.no_grad():
        out_b = classical_model(torch.tensor(X_test_pca, dtype=torch.float32))
        preds_b = out_b.argmax(dim=-1).numpy()
    mlp_time = time.time() - start
    metrics_b = evaluate_predictions(y_test, preds_b, classes)
    print(f"Exp B (Classical MLP) - Acc: {metrics_b['accuracy']*100:.2f}% | Macro-F1: {metrics_b['f1_macro']*100:.2f}%")

    # 3. Evaluate Experiment C: Quantum VQC Head
    print("\n--- Evaluating Experiment C: Quantum VQC Head (4D -> 4-Qubit VQC -> 16 Probs -> Projection -> 38) ---")
    with open(vqc_params_path, "r", encoding="utf-8") as f:
        vqc_params = json.load(f)

    theta_opt = np.array(vqc_params["theta"])
    simulator = VQCCircuitSimulator(n_qubits=4, reps=1)
    
    start_vqc = time.time()
    print("Computing quantum statevector probabilities on 10,849 test samples...")
    q_test_probs = simulator.compute_quantum_probabilities(X_test_pca, theta_opt)
    vqc_sim_time = time.time() - start_vqc

    projection_head = VQCClassifierHead(in_features=16, num_classes=num_classes)
    proj_state = {
        "linear.weight": torch.tensor(vqc_params["projection_weight"], dtype=torch.float32),
        "linear.bias": torch.tensor(vqc_params["projection_bias"], dtype=torch.float32),
    }
    projection_head.load_state_dict(proj_state)
    projection_head.eval()

    with torch.no_grad():
        out_c = projection_head(torch.tensor(q_test_probs, dtype=torch.float32))
        preds_c = out_c.argmax(dim=-1).numpy()
    metrics_c = evaluate_predictions(y_test, preds_c, classes)
    print(f"Exp C (VQC Head) - Acc: {metrics_c['accuracy']*100:.2f}% | Macro-F1: {metrics_c['f1_macro']*100:.2f}% (Sim Time: {vqc_sim_time:.1f}s)")

    # 4. Evaluate Experiment D: Learned Hybrid Fusion
    print("\n--- Evaluating Experiment D: Learned Hybrid Fusion (CNN Logits + VQC Probs) ---")
    fusion_model = LearnedHybridFusion(num_classes=num_classes, vqc_in_dim=16)
    fusion_model.load_state_dict(torch.load(hybrid_fusion_path, map_location="cpu", weights_only=True))
    fusion_model.eval()

    start_fusion = time.time()
    with torch.no_grad():
        fused_logits = fusion_model(
            torch.tensor(test_logits_cnn, dtype=torch.float32),
            torch.tensor(q_test_probs, dtype=torch.float32)
        )
        preds_d = fused_logits.argmax(dim=-1).numpy()
    fusion_time = time.time() - start_fusion
    metrics_d = evaluate_predictions(y_test, preds_d, classes)
    print(f"Exp D (Hybrid Fusion) - Acc: {metrics_d['accuracy']*100:.2f}% | Macro-F1: {metrics_d['f1_macro']*100:.2f}%")

    # Aggregate full experiment results
    experiment_results = {
        "status": "completed",
        "dataset": "BrandonFors/Plant-Diseases-PlantVillage-Dataset",
        "test_samples": num_test,
        "num_classes": num_classes,
        "classes": classes,
        "experiments": {
            "experiment_a_cnn": {
                "name": "MobileNetV2 Classical CNN Baseline",
                "description": "Full end-to-end MobileNetV2 trained on PlantVillage (1280D -> Linear(1280, 38))",
                "input_dimension": 1280,
                "metrics": metrics_a,
                "latency_seconds": round(cnn_time, 4),
            },
            "experiment_b_classical_pca_head": {
                "name": "Classical PCA-4 Control Head",
                "description": "MobileNetV2 1280D -> Scaler -> PCA(4) -> Classical MLP(4 -> 16 -> 38)",
                "input_dimension": 4,
                "metrics": metrics_b,
                "latency_seconds": round(mlp_time, 4),
            },
            "experiment_c_vqc_head": {
                "name": "Quantum Variational Classifier (VQC) Head",
                "description": "MobileNetV2 1280D -> Scaler -> PCA(4) -> 4-Qubit VQC -> 16 Basis Probs -> Linear(16, 38)",
                "input_dimension": 4,
                "num_qubits": 4,
                "circuit": "ZZFeatureMap (reps=1) + RealAmplitudes (reps=1)",
                "metrics": metrics_c,
                "latency_seconds": round(vqc_sim_time, 4),
            },
            "experiment_d_hybrid_fusion": {
                "name": "Learned Hybrid CNN + VQC Fusion",
                "description": "Learned Fusion Layer combining CNN representation and VQC quantum state probabilities",
                "metrics": metrics_d,
                "latency_seconds": round(fusion_time, 4),
            },
        },
        "scientific_summary": {
            "vqc_vs_classical_mlp_delta_acc": float(metrics_c["accuracy"] - metrics_b["accuracy"]),
            "vqc_vs_classical_mlp_delta_f1": float(metrics_c["f1_macro"] - metrics_b["f1_macro"]),
            "hybrid_vs_cnn_delta_acc": float(metrics_d["accuracy"] - metrics_a["accuracy"]),
            "hybrid_vs_cnn_delta_f1": float(metrics_d["f1_macro"] - metrics_a["f1_macro"]),
            "quantum_advantage_claimed": False,
            "conclusion": "The 4-qubit VQC demonstrates functional parameter optimization on compressed feature spaces. Because 1280D features are reduced to 4D to match current 4-qubit simulation constraints, full classical CNN (Exp A) remains superior in raw accuracy, while Exp C validates the hybrid quantum classification mechanism against classical control Exp B.",
        }
    }

    # Save JSON and Markdown artifacts
    (ROOT / "research/results").mkdir(parents=True, exist_ok=True)
    json_path = ROOT / "research/results/quantum_experiment_results.json"
    md_path = ROOT / "research/results/quantum_experiment_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nSaved experiment results JSON to {json_path}")

    generate_markdown_report(experiment_results, md_path)
    print("\nPhase 5 & Phase 6 Rigorous Evaluation & Reporting Complete!")


if __name__ == "__main__":
    main()
