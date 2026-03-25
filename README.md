# QuantumCrop AI: Full Setup Guide (VS Code)

QuantumCrop AI is a Hybrid Quantum-Classical Crop Disease Detection System. This guide will help you set up and run the full-stack application on your local machine using VS Code.

## 1. Prerequisites

Ensure you have the following installed:
- **Node.js** (v18 or higher)
- **npm** (comes with Node.js)
- **Python** (v3.9 or higher)
- **VS Code**

## 2. Project Structure

```text
├── src/                # React Frontend (Vite)
├── research/           # Python Research & Quantum ML
│   ├── models/         # Saved metrics and model states
│   ├── hybrid_pipeline.py  # VQC Simulation & Feature Extraction
│   └── cnn_training.py     # Continuous Learning Engine
├── server.ts           # Express.js Backend & Python Bridge
├── package.json        # Node.js Dependencies
└── .env                # Environment Variables (Create this)
```

## 3. Installation Steps

### Step 1: Clone or Open the Project
Open the project folder in VS Code.

### Step 2: Install Node.js Dependencies
Open the VS Code terminal (`Ctrl+` `) and run:
```bash
npm install
```

### Step 3: Set Up Python Environment
It is recommended to use a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install Required Python Packages
pip install numpy qiskit qiskit-aer pillow
```

## 4. Configuration (.env)

Create a `.env` file in the root directory and add your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENWEATHER_API_KEY=your_openweather_key_here
NODE_ENV=development
```

## 5. Running the Application

### Start the Full-Stack Server
In the VS Code terminal, run:
```bash
npm run dev
```
This command starts the **Express backend** on port 3000 and integrates the **Vite frontend** as middleware.

### Access the App
Open your browser and go to:
`http://localhost:3000`

## 6. Using the Quantum Features

- **Hybrid Diagnostics:** When you upload an image in the "Scanner" tab, the backend automatically calls `research/hybrid_pipeline.py` using a Python bridge.
- **Quantum Hardware Telemetry:** Visible in the "Lab" tab (Validation section), this shows real-time metrics from the quantum simulation, including backend type, qubits, and circuit depth.
- **Quantum State Monitoring:** The dashboard now displays the current **Quantum State** (e.g., "Superposition (Qiskit Statevector)" vs "Simulated (Classical Approximation)"). This is visible in both the Scanner and Lab tabs when not in Farmer Mode.
- **Model Retraining:** In the "Lab" tab, clicking "Start VQC Training" triggers `research/cnn_training.py`, which processes user feedback and updates the simulated quantum decision boundaries.

## 7. Troubleshooting

- **Python Path:** If the server cannot find `python3`, ensure Python is in your system PATH or update the `pyBinary` variable in `server.ts` to point to your specific executable.
- **Qiskit Errors:** If Qiskit fails to install on certain systems, the Python scripts include a robust simulation fallback so the app remains functional.
- **Port 3000:** Ensure no other service is running on port 3000.
