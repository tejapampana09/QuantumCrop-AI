from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from research.training.train_vqc import (
    VQCCircuitSimulator,
    VQCClassifierHead,
    ClassicalPCAControlHead,
    LearnedHybridFusion,
)

ROOT = Path(__file__).resolve().parents[1]


def test_vqc_simulator_circuit():
    sim = VQCCircuitSimulator(n_qubits=4, reps=1)
    assert sim.n_qubits == 4
    assert sim.theta_dim == 8

    # Test single 4D input
    X = np.array([[0.5, -0.2, 1.2, -0.8], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    theta = np.zeros(8, dtype=np.float32)

    probs = sim.compute_quantum_probabilities(X, theta)
    assert probs.shape == (2, 16), f"Expected shape (2, 16), got {probs.shape}"
    assert np.allclose(probs.sum(axis=1), np.array([1.0, 1.0]), atol=1e-5)
    assert (probs >= 0.0).all(), "Negative probabilities encountered"


def test_vqc_trainable_projection_head():
    head = VQCClassifierHead(in_features=16, num_classes=38)
    quantum_probs = torch.rand(4, 16)
    quantum_probs = quantum_probs / quantum_probs.sum(dim=-1, keepdim=True)

    logits = head(quantum_probs)
    assert logits.shape == (4, 38)

    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_classical_pca_control_head():
    classical_mlp = ClassicalPCAControlHead(in_features=4, hidden_dim=16, num_classes=38)
    x = torch.randn(5, 4)
    logits = classical_mlp(x)
    assert logits.shape == (5, 38)


def test_learned_hybrid_fusion():
    fusion = LearnedHybridFusion(num_classes=38, vqc_in_dim=16)
    cnn_logits = torch.randn(3, 38)
    vqc_probs = torch.rand(3, 16)
    vqc_probs = vqc_probs / vqc_probs.sum(dim=-1, keepdim=True)

    fused_out = fusion(cnn_logits, vqc_probs)
    assert fused_out.shape == (3, 38)
