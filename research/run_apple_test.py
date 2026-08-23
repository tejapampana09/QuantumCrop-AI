import sys
sys.path.insert(0, '.')
from research.inference.predict import PredictionPipeline

pipeline = PredictionPipeline()
res = pipeline.predict('research/sample_apple_scab.png')
print('=== PREDICTION ON AUTHENTIC APPLE SCAB SAMPLE ===')
print('CNN Prediction:', res['cnn']['prediction'])
print('CNN Confidence:', f"{res['cnn']['confidence']*100:.2f}%")
print('VQC Prediction:', res['vqc']['prediction'])
print('Hybrid Prediction:', res['hybrid']['prediction'])
