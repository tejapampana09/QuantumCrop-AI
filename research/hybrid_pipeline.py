import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.inference.predict import PredictionPipeline


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python research/hybrid_pipeline.py <image_path>")

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    pipeline = PredictionPipeline()
    result = pipeline.predict(image_path)

    # Return structure matching server.ts requirements
    output = {
        "status": "hybrid_ready" if result["vqc"] is not None else "cnn_only",
        "cnn": result["cnn"],
        "vqc": result["vqc"],
        "hybrid": result["hybrid"],
        "message": (
            "Genuine Hybrid Quantum-Classical Pipeline active. No synthetic quantum prediction."
            if result["vqc"] is not None
            else "Trained CNN active. VQC parameters pending."
        ),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
