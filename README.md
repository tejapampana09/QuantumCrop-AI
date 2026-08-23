# QuantumCrop AI

A hybrid quantum-classical crop disease research platform.

## Important architecture change

The original project contained simulated training metrics and random prediction fallbacks. The rebuilt version removes those claims. A disease prediction is only returned when a real trained MobileNetV2 checkpoint exists.

### ML flow

```text
Labelled crop images
        ↓
70 / 15 / 15 split
        ↓
MobileNetV2 transfer learning
        ↓
Held-out test evaluation
        ↓
CNN feature extraction (1280D)
        ↓
StandardScaler + PCA (4D)
        ↓
Qiskit VQC
ZZFeatureMap + RealAmplitudes
        ↓
COBYLA optimization
        ↓
Hybrid research evaluation
```

## Dataset

The uploaded ZIP does not contain a training dataset. Add a labelled dataset under `research/data/plantvillage/` or pass `--data-dir` to the training script.

## Train

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r research/requirements.txt
python research/training/train_cnn.py --epochs 8
python research/training/extract_features.py
python research/training/train_vqc.py
python research/evaluation/evaluate.py
```

The web app uses the trained CNN for inference. Gemini is an explanation/advisory layer, not the scientific classifier.
