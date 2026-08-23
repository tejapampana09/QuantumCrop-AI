from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, indices=None, transform=None):
        self.dataset = hf_dataset
        self.indices = indices if indices is not None else np.arange(len(hf_dataset))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[int(self.indices[idx])]
        image = item["image"]
        label = item["label"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def extract_split(
    model: nn.Module,
    classifier_layer: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = ""
):
    features_list = []
    logits_list = []
    labels_list = []

    total_batches = len(loader)
    start = time.time()
    print(f"Starting extraction for {desc} ({len(loader.dataset)} samples)...")

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            feat = model(images)  # 1280-dim representation
            logits = classifier_layer(feat)  # 38-dim CNN logits

            features_list.append(feat.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
            labels_list.append(labels.numpy())

            if (i + 1) % 100 == 0 or (i + 1) == total_batches:
                elapsed = time.time() - start
                print(f"  [{desc}] Batch {i+1}/{total_batches} processed ({elapsed:.1f}s)")

    X_feat = np.concatenate(features_list, axis=0)
    X_logits = np.concatenate(logits_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X_feat, X_logits, y


def main():
    parser = argparse.ArgumentParser(description="Extract 1280D CNN features and perform leakage-safe PCA reduction.")
    parser.add_argument("--batch-size", type=int, default=128, help="DataLoader batch size")
    parser.add_argument("--components", type=int, default=4, help="Number of PCA components")
    parser.add_argument("--checkpoint", default="research/models/mobilenetv2_best.pt")
    args = parser.parse_args()

    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        ckpt_path = ROOT / "research/mobilenetv2_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint} or research/mobilenetv2_best.pt")

    print(f"Loading MobileNetV2 checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = checkpoint.get("label_names") or checkpoint.get("classes")
    assert classes is not None, "Classes not found in checkpoint"

    # Build model and load trained state
    full_model = models.mobilenet_v2(weights=None)
    in_features = full_model.classifier[1].in_features
    full_model.classifier[1] = nn.Linear(in_features, len(classes))
    full_model.load_state_dict(checkpoint.get("model_state_dict") or checkpoint.get("state_dict"))
    full_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    full_model = full_model.to(device)

    # Separate feature extractor and classifier head
    feature_extractor = models.mobilenet_v2(weights=None)
    feature_extractor.classifier[1] = nn.Linear(in_features, len(classes))
    feature_extractor.load_state_dict(checkpoint.get("model_state_dict") or checkpoint.get("state_dict"))
    feature_extractor.classifier = nn.Identity()
    feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)

    classifier_head = full_model.classifier.to(device)

    # Standard ImageNet normalization
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    manifest_path = ROOT / "research/split_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dataset_name = manifest["dataset"]
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name)

    train_raw = dataset["train"]
    test_raw = dataset["test"]

    train_indices = np.arange(len(train_raw))
    train_labels = np.array(train_raw["label"])

    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=manifest.get("validation_fraction_from_official_train", 0.15),
        random_state=manifest.get("random_state", 42),
        stratify=train_labels,
    )

    train_ds = HFDatasetWrapper(train_raw, train_idx, transform=eval_tf)
    val_ds = HFDatasetWrapper(train_raw, val_idx, transform=eval_tf)
    test_ds = HFDatasetWrapper(test_raw, None, transform=eval_tf)

    workers = 0 if (device.type == "cpu" or torch.cuda.is_available()) else 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)

    # Extract 1280D features and CNN logits
    X_train_1280, train_logits, y_train = extract_split(feature_extractor, classifier_head, train_loader, device, "TRAIN")
    X_val_1280, val_logits, y_val = extract_split(feature_extractor, classifier_head, val_loader, device, "VAL")
    X_test_1280, test_logits, y_test = extract_split(feature_extractor, classifier_head, test_loader, device, "TEST")

    assert X_train_1280.shape[1] == 1280, f"Expected 1280 features, got {X_train_1280.shape[1]}"
    assert len(X_train_1280) == manifest["train_samples"], f"Train count mismatch: {len(X_train_1280)} vs {manifest['train_samples']}"
    assert len(X_val_1280) == manifest["validation_samples"], f"Val count mismatch: {len(X_val_1280)} vs {manifest['validation_samples']}"
    assert len(X_test_1280) == manifest["official_test_samples"], f"Test count mismatch: {len(X_test_1280)} vs {manifest['official_test_samples']}"

    # Save 1280D features
    (ROOT / "research/artifacts").mkdir(parents=True, exist_ok=True)
    raw_features_path = ROOT / "research/artifacts/cnn_features_1280.npz"
    np.savez_compressed(
        raw_features_path,
        X_train=X_train_1280,
        y_train=y_train,
        train_logits=train_logits,
        X_val=X_val_1280,
        y_val=y_val,
        val_logits=val_logits,
        X_test=X_test_1280,
        y_test=y_test,
        test_logits=test_logits,
        classes=np.array(classes),
    )
    print(f"Saved 1280D features to {raw_features_path}")

    # Phase 3: Leakage-Safe Preprocessing
    print("\nFitting StandardScaler strictly on training set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_1280)
    X_val_scaled = scaler.transform(X_val_1280)
    X_test_scaled = scaler.transform(X_test_1280)

    print(f"Fitting PCA(n_components={args.components}, random_state=42) strictly on training set...")
    pca = PCA(n_components=args.components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Save Scaler & PCA
    (ROOT / "research/models").mkdir(parents=True, exist_ok=True)
    scaler_path_models = ROOT / "research/models/feature_scaler.joblib"
    pca_path_models = ROOT / "research/models/feature_pca.joblib"
    scaler_path_art = ROOT / "research/artifacts/feature_scaler.joblib"
    pca_path_art = ROOT / "research/artifacts/feature_pca.joblib"

    joblib.dump(scaler, scaler_path_models)
    joblib.dump(pca, pca_path_models)
    joblib.dump(scaler, scaler_path_art)
    joblib.dump(pca, pca_path_art)

    # Save PCA 4D features
    pca_features_path = ROOT / "research/artifacts/cnn_features_pca4.npz"
    np.savez_compressed(
        pca_features_path,
        X_train=X_train_pca,
        y_train=y_train,
        train_logits=train_logits,
        X_val=X_val_pca,
        y_val=y_val,
        val_logits=val_logits,
        X_test=X_test_pca,
        y_test=y_test,
        test_logits=test_logits,
        classes=np.array(classes),
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    print(f"PCA explained variance ratio (4 components): {pca.explained_variance_ratio_} (Sum: {pca.explained_variance_ratio_.sum():.4f})")
    print(f"Saved PCA 4D features to {pca_features_path}")
    print("CNN Feature Extraction & Leakage-Safe Preprocessing Complete!")


if __name__ == "__main__":
    main()
