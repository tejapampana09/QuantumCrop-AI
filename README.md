# QuantumCrop AI

A production-grade **Hybrid Quantum-Classical Agricultural Pathology & Crop Disease Diagnostic Platform**.

---

## 🌟 System Overview

QuantumCrop AI combines a high-efficiency **MobileNetV2 (1280-D)** deep convolutional neural network with an **IBM Qiskit 4-Qubit Variational Quantum Circuit (VQC)**, a **Learned Hybrid Fusion Layer**, and a **Gemini 3.5 Flash Agronomic Advisory Engine**.

```text
                        [ Leaf Image Upload ]
                                  │
                                  ▼
               [ Leaf Isolation & Quality Preprocessing ]
                                  │
                                  ▼
                       [ MobileNetV2 (1280-D) ]
                      Primary Disease Classifier
                       (99.50% Test Accuracy)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       [ Classical PCA-4 Head ]          [ StandardScaler + PCA (4D) ]
             (Exp B: 69.23%)                      │
                                                  ▼
                                       [ 4-Qubit IBM Qiskit VQC ]
                                       ZZFeatureMap + RealAmplitudes
                                       (16 Quantum Basis Probabilities)
                                       (Exp C: 9.87% - 320x Bottleneck)
                                                  │
                  ┌───────────────────────────────┘
                  ▼
       [ Learned Hybrid Fusion MLP ]
              (Exp D: 99.42%)
                  │
                  ▼
       [ Crop Taxonomy Gate & Safety Interlock ]
         (14-Crop Marginals, 0% Wrong-Confident Rate)
                  │
                  ▼
       [ Gemini 3.5 Flash Pathology Advisory ]
         (Organic/Chemical Dosages & 7-Day Recovery)
```

---

## 📊 Measured 4-Experiment Benchmark Results

Evaluated on the official held-out **PlantVillage Test Set (10,849 samples)**:

| Experiment | Architecture | Test Accuracy | Macro F1-Score | Role & Scientific Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Exp A** | **MobileNetV2 (1280D)** | **99.50%** | **0.9950** | **Primary Production Classifier** |
| **Exp B** | Classical PCA-4 Control | 69.23% | 0.5259 | Classical linear projection benchmark |
| **Exp C** | 4-Qubit Qiskit VQC | 9.87% | 0.0047 | Quantum circuit ($16 \to 38$) under 320x PCA compression |
| **Exp D** | Learned Hybrid Fusion MLP | **99.42%** | **0.9896** | Joint representation fusion of CNN + Quantum states |

---

## 🛡️ Real-World Robustness & Safety Interlock

To eliminate the "Lab-to-Field" domain shift and prevent dangerous misdiagnoses on wild outdoor photos:
1. **Zero-Dependency Leaf Saliency Detector** (`research/inference/leaf_detector.py`): Isolates leaf contours from outdoor background clutter.
2. **Deterministic 14-Crop Taxonomy Gate** (`research/inference/crop_gate.py`): Calculates crop marginal probabilities $\sum_{c \in \text{crop}} P(c)$ across all 38 classes.
3. **Multimodal Disagreement Arbiter** (`research/inference/arbitration.py`): Safely flags domain mismatches (`status: crop_mismatch`) to achieve a **0.0% Wrong-Confident Rate**.

---

## ⚡ Quickstart & Installation

### 1. Python Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows (or source .venv/bin/activate on Linux/macOS)

# Install research dependencies
pip install -r research/requirements.txt
```

### 2. Node.js Frontend Setup
```bash
# Install frontend dependencies
npm install

# Configure environment variables
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### 3. Launch Development Server & Inference Daemon
```bash
# Terminal 1: Start warm Python inference daemon (Port 5001)
python research/inference_server.py 5001

# Terminal 2: Start Express + Vite Frontend (Port 3000)
npm run dev
```

Visit **[http://localhost:3000](http://localhost:3000)** to launch the application.

---

## 🧪 Testing & Verification

Run the full automated test suite (24 unit & integration tests):
```bash
pytest -v tests/
```

Compile the production frontend build:
```bash
npm run build
```
