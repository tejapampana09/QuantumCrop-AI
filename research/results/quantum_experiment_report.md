# QuantumCrop AI — Quantum-Classical Experiment Scientific Report

**Date**: 2026-08-23  
**Dataset**: `BrandonFors/Plant-Diseases-PlantVillage-Dataset` (Hugging Face)  
**Untouched Test Set Size**: 10849 samples across 38 crop disease classes  

## 1. Benchmarking Results

| Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Input Dimension | Classifier Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: MobileNetV2 CNN Baseline** | **99.50%** | **99.38%** | **98.83%** | **99.06%** | 1280D | Classical Linear (1280 $	o$ 38) |
| **Exp B: Classical PCA Control Head** | 69.23% | 56.81% | 53.79% | 52.59% | 4D (PCA) | Classical MLP (4 $	o$ 16 $	o$ 38) |
| **Exp C: Quantum VQC Head** | 9.87% | 0.26% | 2.63% | 0.47% | 4D (PCA) | 4-Qubit VQC + Linear (16 $	o$ 38) |
| **Exp D: Learned Hybrid Fusion** | 99.42% | 99.15% | 98.80% | 98.96% | 1280D + 16D | Learned Fusion Layer |

## 2. Scientific Findings & Conclusion
- **Fair Classical Comparison**: Exp B provides direct control for Exp C under identical 4D feature compression.
- **Honest Disclosure**: Full 1280D classical CNN remains superior due to avoiding extreme 320x compression. No unsupported quantum advantage is claimed.
