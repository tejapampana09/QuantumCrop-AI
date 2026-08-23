from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
IMAGE_SIZE = 224


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    for p in model.features.parameters():
        p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def make_transforms():
    weights = models.MobileNet_V2_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=g).tolist()
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(y.tolist())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="research/data/plantvillage")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default="research/models/mobilenetv2_best.pt")
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Dataset not found: {data_dir}. Add a labelled dataset before training.")

    train_tf, eval_tf = make_transforms()
    base = datasets.ImageFolder(data_dir)
    if len(base) < 20 or len(base.classes) < 2:
        raise SystemExit("Dataset is too small. Need at least 20 images and 2 classes.")

    train_idx, val_idx, test_idx = split_indices(len(base), 0.70, 0.15, args.seed)
    train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_tf), train_idx)
    val_ds = Subset(datasets.ImageFolder(data_dir, transform=eval_tf), val_idx)
    test_ds = Subset(datasets.ImageFolder(data_dir, transform=eval_tf), test_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 0 if device.type == "cpu" else 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)

    model = build_model(len(base.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = -1.0
    history = []
    started = time.time()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_ds),
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            torch.save({
                "state_dict": model.state_dict(),
                "classes": base.classes,
                "image_size": IMAGE_SIZE,
                "mean": models.MobileNet_V2_Weights.DEFAULT.transforms().mean,
                "std": models.MobileNet_V2_Weights.DEFAULT.transforms().std,
                "seed": args.seed,
            }, output)

    checkpoint = torch.load(output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    Path("research/models/split_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "train_indices": train_idx,
        "validation_indices": val_idx,
        "test_indices": test_idx
    }, indent=2), encoding="utf-8")
    metadata = {
        "status": "trained",
        "model": "MobileNetV2",
        "dataset_dir": str(data_dir),
        "num_classes": len(base.classes),
        "classes": base.classes,
        "samples": {"total": len(base), "train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)},
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "device": str(device),
        "epochs": args.epochs,
        "history": history,
        "test_metrics": test_metrics,
        "training_seconds": round(time.time() - started, 2),
        "seed": args.seed,
        "checkpoint": str(output),
    }
    Path("research/models").mkdir(exist_ok=True)
    Path("research/models/cnn_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
