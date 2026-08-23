from datasets import load_dataset
import torch

print("Loading 1 sample of Apple Scab from PlantVillage dataset...")
ds = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset", split="train", streaming=True)
for item in ds:
    if item["label"] == 0: # 0 is Apple___Apple_scab
        img = item["image"]
        img.save("research/sample_apple_scab.png")
        print("Saved sample_apple_scab.png!")
        break

from research.inference.predict import PredictionPipeline
pipeline = PredictionPipeline()
res = pipeline.predict("research/sample_apple_scab.png")
print("PREDICTION ON APPLE SCAB SAMPLE:")
print("CNN Prediction:", res["cnn"]["prediction"])
print("CNN Confidence:", f"{res['cnn']['confidence']*100:.2f}%")
