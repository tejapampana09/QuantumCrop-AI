# QuantumCrop AI — Research Architecture

## Ground truth
The previous implementation simulated training, generated random confidences/heatmaps, and hard-coded validation metrics. Those paths have been removed.

## Pipeline

1. Labelled image dataset → deterministic 70/15/15 split
2. MobileNetV2 transfer learning
3. Best checkpoint saved as `research/models/mobilenetv2_best.pt`
4. CNN feature extraction
5. Standardization + PCA to 4 quantum features
6. VQC using Qiskit `ZZFeatureMap + RealAmplitudes`
7. COBYLA optimization
8. Final evaluation on a held-out test set
9. Gemini is used only for explanation/advisory, not as the ground-truth classifier

## Commands

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r research/requirements.txt

# Put labelled images under research/data/plantvillage/
python research/training/train_cnn.py --epochs 8
python research/training/extract_features.py --split train --feature-output research/artifacts/cnn_features.npz
python research/training/extract_features.py --split test --feature-output research/artifacts/cnn_features_test.npz
python research/training/train_vqc.py
python research/evaluation/evaluate.py
```

For a defensible research paper, the VQC must also be evaluated using a held-out split rather than the complete feature set. That is the next refinement after the first successful end-to-end training run.
