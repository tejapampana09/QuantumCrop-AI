import sys
sys.path.insert(0, '.')
from research.inference.predict import PredictionPipeline

pipeline = PredictionPipeline()
img_path = r'C:\Users\tejap\.gemini\antigravity\brain\0dce3af8-ab4d-4edb-955b-4fba2d3bb736\.user_uploaded\media_1787511775311.png'
res = pipeline.predict(img_path)

print('=== PREDICTION ON USER UPLOADED LEAF ===')
print('CNN Prediction:', res['cnn']['prediction'])
print('CNN Confidence:', f"{res['cnn']['confidence']*100:.2f}%")
print('VQC Prediction:', res['vqc']['prediction'])
print('Hybrid Prediction:', res['hybrid']['prediction'])

top3 = sorted(res['cnn']['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
print('\nTop 3 CNN Classes:')
for c, p in top3:
    print(f' - {c}: {p*100:.2f}%')
