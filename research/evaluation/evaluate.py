from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='research/data/plantvillage')
    p.add_argument('--checkpoint', default='research/models/mobilenetv2_best.pt')
    p.add_argument('--output', default='research/models/evaluation.json')
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(ckpt['classes']))
    model.load_state_dict(ckpt['state_dict']); model.eval()
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(ckpt['mean'], ckpt['std'])])
    ds = datasets.ImageFolder(args.data_dir, transform=tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            y_true.extend(y.tolist()); y_pred.extend(model(x).argmax(1).tolist())
    report = classification_report(y_true, y_pred, target_names=ds.classes, output_dict=True, zero_division=0)
    result = {'status':'evaluated', 'classes': ds.classes, 'classification_report': report, 'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
