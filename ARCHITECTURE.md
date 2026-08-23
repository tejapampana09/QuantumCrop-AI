# QuantumCrop AI — Complete System Architecture

---

## 🏛️ System Overview

QuantumCrop AI is designed as a **layered, hybrid quantum-classical diagnostic platform**. It avoids the pitfalls of black-box AI by enforcing a clear separation of concerns between:
1. **Primary High-Efficiency Edge Inference** (MobileNetV2 1280D).
2. **Quantum State Feature Analysis** (IBM Qiskit 4-Qubit VQC).
3. **Learned Representation Fusion** (PyTorch MLP).
4. **Safety & Domain Robustness Gating** (Leaf Saliency + 14-Crop Taxonomy Marginal Gate).
5. **AI Agronomic Advisory** (Google Gemini 3.5 Flash).

---

## 🔬 End-to-End Dataflow

```text
                     [ INPUT LEAF PHOTOGRAPH ]
                                │
                                ▼
         [ 1. Zero-Dependency Leaf Saliency & BBox Extractor ]
               (Otsu Thresholding + Excess Green Index)
                                │
                                ▼
         [ 2. MobileNetV2 (1280-D) Deep Feature Extractor ]
             ├── Produces 1280-D spatial feature tensor
             └── Primary Classification Head (38 Classes)
                                │
               ┌────────────────┴────────────────┐
               │                                 │
               ▼                                 ▼
      [ Classical Logits ]             [ StandardScaler (Train-only) ]
     (MobileNetV2: 99.50%)                       │
               │                                 ▼
               │                        [ PCA (1280D -> 4D) ]
               │                        (320x Feature Compression)
               │                                 │
               │                                 ▼
               │                      [ 4-Qubit IBM Qiskit VQC ]
               │                      - ZZFeatureMap (reps=1)
               │                      - RealAmplitudes Ansatz (reps=1)
               │                      - Statevector Simulator
               │                                 │
               │                                 ▼
               │                      [ 16 Quantum Probabilities ]
               │                                 │
               │                                 ▼
               │                      [ Linear Projection (16->38) ]
               │                             (VQC: 9.87%)
               │                                 │
               └────────────────┬────────────────┘
                                │
                                ▼
         [ 3. Learned Hybrid Representation Fusion MLP ]
             ├── Input: [CNN Logits (38) + Quantum Probabilities (16)]
             └── Output: 38-Class Fused Distribution (99.42% Test Acc)
                                │
                                ▼
         [ 4. Crop Taxonomy Gate & Safety Interlock ]
             ├── Computes 14-Crop Marginal Probabilities: Sum_{c in Crop} P(c)
             └── Detects Ambiguity & Intercepts Domain Shifts (0% Wrong-Confident)
                                │
                                ▼
         [ 5. Gemini 3.5 Flash Agronomic Advisory Layer ]
             ├── Secondary visual cross-verification
             ├── Disease progression & pathological symptoms breakdown
             ├── Precise Organic & Chemical remedies with dosages
             └── 7-Day Actionable Farmer Recovery Plan
```

---

## 📋 Role of System Components

| Component | Architecture | Measured Accuracy | System Role |
| :--- | :--- | :--- | :--- |
| **MobileNetV2** | 1280-D CNN | **99.50%** | **Primary Production Classifier** (Provides authoritative diagnosis). |
| **PCA-4 Control** | Linear 4D Head | **69.23%** | **Classical Baseline Control** (Quantifies the information ceiling of 4D inputs). |
| **4-Qubit VQC** | Qiskit VQC ($16 \to 38$) | **9.87%** | **Experimental Quantum Layer** (Encodes PCA features into quantum entanglement). |
| **Learned Hybrid** | PyTorch MLP | **99.42%** | **Supplementary Fusion** (Evaluates joint classical + quantum consensus). |
| **Crop Gate** | 14-Crop Taxonomy | **100.0%** | **Safety Interlock** (Prevents cross-crop misclassification). |
| **Gemini 3.5 Flash** | Multimodal LLM | **Advisory** | **Agronomic Explanation Layer** (Actionable dosages and farmer advice). |

---

## 🔒 Architectural Integrity Guarantees

1. **MobileNetV2 is Primary**: The displayed primary diagnosis is derived from the trained MobileNetV2 classifier. Gemini acts as an advisory and cross-check layer and **does not silently overwrite** the primary model's prediction.
2. **Zero Leakage**: Scaler and PCA artifacts were fitted exclusively on the 70% training split (`split_manifest.json`), preserving zero data leakage to validation and test sets.
3. **Sub-Second Runtime**: The in-memory Python daemon (`research/inference_server.py`) keeps PyTorch and Qiskit circuits preloaded, delivering ~48ms warm inference.
