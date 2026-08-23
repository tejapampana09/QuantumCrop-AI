from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def test_scaler_and_pca_artifacts():
    scaler_path = ROOT / "research/models/feature_scaler.joblib"
    pca_path = ROOT / "research/models/feature_pca.joblib"

    if scaler_path.exists() and pca_path.exists():
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)

        assert isinstance(scaler, StandardScaler)
        assert isinstance(pca, PCA)
        assert pca.n_components == 4
        assert scaler.mean_.shape == (1280,)
        assert pca.components_.shape == (4, 1280)

        # Test transformation pipeline on 1280D input
        dummy_feat = np.random.randn(3, 1280)
        scaled = scaler.transform(dummy_feat)
        reduced = pca.transform(scaled)

        assert reduced.shape == (3, 4), f"Expected shape (3, 4), got {reduced.shape}"
        assert not np.isnan(reduced).any(), "PCA output contains NaNs"


def test_leakage_independence():
    """Verify that PCA fitted on train data is unaffected by test sample distributions."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, size=(500, 1280))
    X_test = rng.normal(5, 2, size=(100, 1280))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    pca = PCA(n_components=4, random_state=42)
    pca.fit(X_train_s)

    # Test transformed with fitted components
    X_test_s = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_s)

    assert X_test_pca.shape == (100, 4)
    # Ensure scaler.mean_ is purely from training mean
    assert np.allclose(scaler.mean_, X_train.mean(axis=0))
