from __future__ import annotations

from pathlib import Path
import pytest
import torch
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]


def test_mobilenetv2_feature_extraction():
    ckpt_path = ROOT / "research/models/mobilenetv2_best.pt"
    if not ckpt_path.exists():
        ckpt_path = ROOT / "research/mobilenetv2_best.pt"

    assert ckpt_path.exists(), "MobileNetV2 checkpoint must exist"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    classes = checkpoint.get("label_names") or checkpoint.get("classes")
    num_classes = len(classes)
    assert num_classes == 38

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    assert in_features == 1280

    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint.get("model_state_dict") or checkpoint.get("state_dict"))
    model.eval()

    # Bypass classifier
    model.classifier = nn.Identity()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 1280), f"Expected shape (2, 1280), got {out.shape}"
    assert not torch.isnan(out).any(), "Output features contains NaNs"
