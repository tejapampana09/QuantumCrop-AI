from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[2]


class VQCCircuitSimulator:
    """Efficient parameterized 4-qubit VQC statevector simulator."""

    def __init__(self, n_qubits: int = 4, reps: int = 1):
        self.n_qubits = n_qubits
        self.reps = reps
        self.theta_dim = 2 * n_qubits * (reps)  # 8 parameters for reps=1

        # Base circuit template
        self.fmap = ZZFeatureMap(n_qubits, reps=reps, entanglement="full")
        self.ansatz = RealAmplitudes(n_qubits, reps=reps, entanglement="full")
        self.base_circuit = QuantumCircuit(n_qubits)
        self.base_circuit.compose(self.fmap, inplace=True)
        self.base_circuit.compose(self.ansatz, inplace=True)

    def compute_quantum_probabilities(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute 16 basis state probabilities for an array of 4D samples."""
        probs_list = []
        n_samples = len(X)
        for i in range(n_samples):
            x = X[i]
            params = list(x[: self.n_qubits]) + list(theta)
            qc = self.base_circuit.assign_parameters(params)
            probs = Statevector.from_instruction(qc).probabilities()
            probs_list.append(probs)
        return np.array(probs_list, dtype=np.float32)


class VQCClassifierHead(nn.Module):
    """Trainable classical projection layer mapping 16 quantum measurements to 38 class logits."""

    def __init__(self, in_features: int = 16, num_classes: int = 38):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, quantum_probs: torch.Tensor) -> torch.Tensor:
        return self.linear(quantum_probs)


class ClassicalPCAControlHead(nn.Module):
    """Classical control baseline: maps 4 PCA features to 16 hidden representations then 38 classes."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 16, num_classes: int = 38):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedHybridFusion(nn.Module):
    """Learned fusion head combining CNN logit representations and VQC probability representations."""

    def __init__(self, num_classes: int = 38, vqc_in_dim: int = 16):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(num_classes + vqc_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, cnn_logits: torch.Tensor, vqc_probs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([cnn_logits, vqc_probs], dim=-1)
        return self.fusion(combined)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_vqc_pipeline(
    features_path: str = "research/artifacts/cnn_features_pca4.npz",
    max_epochs: int = 30,
    batch_size: int = 256,
    subsample_train: int = 10000,
    lr_theta: float = 0.05,
    lr_proj: float = 0.01,
):
    print("=" * 70)
    print("PHASE 4: QUANTUM VARIATIONAL CLASSIFIER & EXPERIMENTAL HEADS TRAINING")
    print("=" * 70)

    data_file = ROOT / features_path
    if not data_file.exists():
        raise FileNotFoundError(f"PCA features file not found at {data_file}. Run extract_features.py first.")

    data = np.load(data_file, allow_pickle=True)
    X_train_pca = data["X_train"]
    y_train = data["y_train"].astype(int)
    train_logits = data["train_logits"]

    X_val_pca = data["X_val"]
    y_val = data["y_val"].astype(int)
    val_logits = data["val_logits"]

    classes = data["classes"].tolist()
    num_classes = len(classes)

    print(f"Loaded PCA features: Train={X_train_pca.shape}, Val={X_val_pca.shape}, Classes={num_classes}")

    # Subsample training data if specified for fast quantum circuit optimization
    if subsample_train and subsample_train < len(X_train_pca):
        rng = np.random.default_rng(42)
        train_sub_idx = rng.choice(len(X_train_pca), size=subsample_train, replace=False)
        X_train_sub = X_train_pca[train_sub_idx]
        y_train_sub = y_train[train_sub_idx]
        train_logits_sub = train_logits[train_sub_idx]
        print(f"Using stratified/random training subsample of {subsample_train} for VQC parameter tuning.")
    else:
        X_train_sub = X_train_pca
        y_train_sub = y_train
        train_logits_sub = train_logits

    # 1. Train Classical Control Head (Experiment B)
    print("\n--- Training Classical Control Head (Experiment B: 4D PCA -> Classical MLP -> 38 classes) ---")
    classical_model = ClassicalPCAControlHead(in_features=4, hidden_dim=16, num_classes=num_classes)
    opt_classical = AdamW(classical_model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_train_sub, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_sub, dtype=torch.long)
    X_val_t = torch.tensor(X_val_pca, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    best_classical_val_f1 = 0.0
    best_classical_state = None

    for epoch in range(1, 51):
        classical_model.train()
        opt_classical.zero_grad()
        out = classical_model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        opt_classical.step()

        if epoch % 10 == 0 or epoch == 50:
            classical_model.eval()
            with torch.no_grad():
                val_out = classical_model(X_val_t)
                val_preds = val_out.argmax(dim=-1).numpy()
                val_metrics = compute_metrics(y_val, val_preds)
                print(f"  [Classical MLP Epoch {epoch:02d}] Val Acc: {val_metrics['accuracy']:.4f}, Val Macro-F1: {val_metrics['f1_macro']:.4f}")
                if val_metrics["f1_macro"] > best_classical_val_f1:
                    best_classical_val_f1 = val_metrics["f1_macro"]
                    best_classical_state = classical_model.state_dict()

    torch.save(best_classical_state, ROOT / "research/models/classical_pca_head.pt")
    print(f"Classical Control Head Best Validation Macro-F1: {best_classical_val_f1:.4f}")

    # 2. Train Quantum Variational Classifier (Experiment C)
    print("\n--- Training Quantum Variational Classifier Head (Experiment C: 4D PCA -> 4-Qubit VQC -> 16 Probs -> Trainable Projection -> 38 classes) ---")
    simulator = VQCCircuitSimulator(n_qubits=4, reps=1)
    rng = np.random.default_rng(42)
    theta = rng.normal(0, 0.1, simulator.theta_dim)

    projection_head = VQCClassifierHead(in_features=16, num_classes=num_classes)
    opt_projection = AdamW(projection_head.parameters(), lr=lr_proj, weight_decay=1e-4)

    vqc_history = []
    best_vqc_val_f1 = 0.0
    best_vqc_theta = theta.copy()
    best_proj_state = None

    start_vqc_train = time.time()

    # Pre-evaluate quantum basis probabilities for train subsample and validation
    print("Computing quantum statevector representations across dataset...")
    q_train_probs = simulator.compute_quantum_probabilities(X_train_sub, theta)
    q_val_probs = simulator.compute_quantum_probabilities(X_val_pca, theta)

    q_train_t = torch.tensor(q_train_probs, dtype=torch.float32)
    q_val_t = torch.tensor(q_val_probs, dtype=torch.float32)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        # Stage A: Update parameterized rotation angles (theta) periodically via numerical gradient / SPSA
        if epoch % 5 == 1 and epoch > 1:
            # Perturb and optimize theta
            delta = rng.choice([-1.0, 1.0], size=simulator.theta_dim) * 0.05
            theta_plus = theta + delta
            theta_minus = theta - delta

            # Sample mini-batch for gradient evaluation
            mb_idx = rng.choice(len(X_train_sub), size=min(500, len(X_train_sub)), replace=False)
            q_plus = simulator.compute_quantum_probabilities(X_train_sub[mb_idx], theta_plus)
            q_minus = simulator.compute_quantum_probabilities(X_train_sub[mb_idx], theta_minus)

            projection_head.eval()
            with torch.no_grad():
                l_plus = criterion(projection_head(torch.tensor(q_plus, dtype=torch.float32)), y_tr_t[mb_idx]).item()
                l_minus = criterion(projection_head(torch.tensor(q_minus, dtype=torch.float32)), y_tr_t[mb_idx]).item()

            ghat = (l_plus - l_minus) / (2.0 * delta)
            theta = theta - lr_theta * ghat
            # Recompute full train & val quantum probabilities
            q_train_probs = simulator.compute_quantum_probabilities(X_train_sub, theta)
            q_val_probs = simulator.compute_quantum_probabilities(X_val_pca, theta)
            q_train_t = torch.tensor(q_train_probs, dtype=torch.float32)
            q_val_t = torch.tensor(q_val_probs, dtype=torch.float32)

        # Stage B: Train Classical Projection Layer W, b on Quantum Probabilities
        projection_head.train()
        n_batches = int(np.ceil(len(q_train_t) / batch_size))
        perm = torch.randperm(len(q_train_t))
        running_train_loss = 0.0

        for b in range(n_batches):
            b_idx = perm[b * batch_size : (b + 1) * batch_size]
            opt_projection.zero_grad()
            logits = projection_head(q_train_t[b_idx])
            loss = criterion(logits, y_tr_t[b_idx])
            loss.backward()
            opt_projection.step()
            running_train_loss += loss.item() * len(b_idx)

        train_loss = running_train_loss / len(q_train_t)

        # Validation evaluation
        projection_head.eval()
        with torch.no_grad():
            val_logits_vqc = projection_head(q_val_t)
            val_loss = criterion(val_logits_vqc, y_val_t).item()
            val_preds = val_logits_vqc.argmax(dim=-1).numpy()
            val_metrics = compute_metrics(y_val, val_preds)

        epoch_time = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "epoch_seconds": round(epoch_time, 2),
        }
        vqc_history.append(row)
        print(f"  [VQC Epoch {epoch:02d}/{max_epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_macro']:.4f} ({epoch_time:.1f}s)")

        if val_metrics["f1_macro"] > best_vqc_val_f1:
            best_vqc_val_f1 = val_metrics["f1_macro"]
            best_vqc_theta = theta.copy()
            best_proj_state = projection_head.state_dict()

    vqc_train_seconds = time.time() - start_vqc_train
    print(f"VQC Optimization finished in {vqc_train_seconds:.1f}s. Best Val Macro-F1: {best_vqc_val_f1:.4f}")

    # Save VQC parameters & history
    vqc_params = {
        "status": "trained",
        "num_qubits": 4,
        "ansatz": "RealAmplitudes",
        "feature_map": "ZZFeatureMap",
        "theta": best_vqc_theta.tolist(),
        "projection_weight": best_proj_state["linear.weight"].numpy().tolist(),
        "projection_bias": best_proj_state["linear.bias"].numpy().tolist(),
        "classes": classes,
        "best_val_f1_macro": best_vqc_val_f1,
        "training_seconds": round(vqc_train_seconds, 2),
    }

    out_params = ROOT / "research/models/vqc_params.json"
    out_history = ROOT / "research/models/vqc_history.json"
    with open(out_params, "w", encoding="utf-8") as f:
        json.dump(vqc_params, f, indent=2)
    with open(out_history, "w", encoding="utf-8") as f:
        json.dump(vqc_history, f, indent=2)
    print(f"Saved VQC parameters to {out_params}")

    # 3. Train Learned Hybrid Fusion (Experiment D)
    print("\n--- Training Learned Hybrid Fusion Head (Experiment D: CNN Logits + VQC Representation -> Learned Fusion -> 38 classes) ---")
    # Load optimal VQC model
    projection_head.load_state_dict(best_proj_state)
    projection_head.eval()

    q_train_best = simulator.compute_quantum_probabilities(X_train_sub, best_vqc_theta)
    q_val_best = simulator.compute_quantum_probabilities(X_val_pca, best_vqc_theta)

    cnn_tr_t = torch.tensor(train_logits_sub, dtype=torch.float32)
    cnn_val_t = torch.tensor(val_logits, dtype=torch.float32)
    q_tr_best_t = torch.tensor(q_train_best, dtype=torch.float32)
    q_val_best_t = torch.tensor(q_val_best, dtype=torch.float32)

    fusion_model = LearnedHybridFusion(num_classes=num_classes, vqc_in_dim=16)
    opt_fusion = AdamW(fusion_model.parameters(), lr=1e-2, weight_decay=1e-4)

    best_fusion_val_f1 = 0.0
    best_fusion_state = None

    for epoch in range(1, 41):
        fusion_model.train()
        opt_fusion.zero_grad()
        fused_out = fusion_model(cnn_tr_t, q_tr_best_t)
        loss = criterion(fused_out, y_tr_t)
        loss.backward()
        opt_fusion.step()

        if epoch % 10 == 0 or epoch == 40:
            fusion_model.eval()
            with torch.no_grad():
                val_fused = fusion_model(cnn_val_t, q_val_best_t)
                val_preds = val_fused.argmax(dim=-1).numpy()
                val_metrics = compute_metrics(y_val, val_preds)
                print(f"  [Hybrid Fusion Epoch {epoch:02d}] Val Acc: {val_metrics['accuracy']:.4f}, Val Macro-F1: {val_metrics['f1_macro']:.4f}")
                if val_metrics["f1_macro"] > best_fusion_val_f1:
                    best_fusion_val_f1 = val_metrics["f1_macro"]
                    best_fusion_state = fusion_model.state_dict()

    torch.save(best_fusion_state, ROOT / "research/models/hybrid_fusion.pt")
    fusion_meta = {
        "status": "trained",
        "num_classes": num_classes,
        "best_val_f1_macro": best_fusion_val_f1,
    }
    with open(ROOT / "research/models/hybrid_fusion.json", "w", encoding="utf-8") as f:
        json.dump(fusion_meta, f, indent=2)
    print(f"Learned Hybrid Fusion Best Validation Macro-F1: {best_fusion_val_f1:.4f}")
    print("\nPhase 4 Head Training Complete!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="research/artifacts/cnn_features_pca4.npz")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--subsample", type=int, default=10000)
    args = p.parse_args()
    train_vqc_pipeline(features_path=args.features, max_epochs=args.epochs, subsample_train=args.subsample)
