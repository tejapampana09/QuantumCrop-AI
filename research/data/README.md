# Dataset

The uploaded project does **not** contain a training dataset. Put the dataset here (or pass `--data-dir`):

```text
data/plantvillage/
├── Class_A/
│   ├── image1.jpg
│   └── ...
├── Class_B/
├── Class_C/
└── ...
```

The training pipeline discovers class directories automatically. It creates deterministic train/validation/test splits and never generates synthetic labels or fake metrics.

Recommended first experiment: use a small, documented subset of PlantVillage so the project is reproducible on CPU. Keep a separate untouched test set for final reporting.
