# QuantumCrop AI — Rebuilt Architecture

## Current truth

The original ZIP did not contain a training dataset and the research scripts did not train a CNN or VQC. They generated synthetic metrics, random confidences and mock heatmaps. The rebuild removes those paths.

## End-to-end design

```text
                LABELLED DATASET
                       │
                       ▼
              Deterministic Split
              70% / 15% / 15%
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          TRAIN              HOLD-OUT
             │              VAL + TEST
             ▼                   │
       MobileNetV2               │
       Transfer Learning         │
             │                   │
             ▼                   │
       Best Checkpoint           │
             │                   │
             ▼                   │
      1280-D CNN Features        │
             │                   │
             ▼                   │
     Train-only StandardScaler   │
             │                   │
             ▼                   │
        Train-only PCA           │
             │                   │
             ▼                   │
        4-D Quantum Input        │
             │                   │
             ▼                   │
    ZZFeatureMap + RealAmplitudes│
             │                   │
             ▼                   │
           COBYLA                │
             │                   │
             ▼                   │
       Trained VQC               │
             │                   │
             └──────────┬────────┘
                        ▼
              Held-out evaluation
                        │
                        ▼
        CNN vs VQC vs Hybrid report
```

## Runtime

The web API calls the trained MobileNetV2 checkpoint. If it is not present, `/api/predict` returns a clear training-required error instead of inventing a prediction.

Gemini is downstream of the model: it can explain a model result, translate advice, and provide contextual information. It is not used to manufacture model accuracy.

## Next research stage

After the first real CNN run succeeds, the VQC should be evaluated on the same untouched test split using the scaler/PCA fitted only on training data. Only then should the project claim a measured hybrid result.
