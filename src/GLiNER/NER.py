import os
from datetime import datetime
import torch

print("Set environment variables for offline mode", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("Load GLiNER Entity Prediction", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
from gliner.model import GLiNER

# Load the model from the saved directory
print("Loading model...", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = GLiNER.from_pretrained("models/checkpoint-5900").to(device)
print(f"Model on {device} loaded successfully", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

text = "Gessner bittet Bullinger um die Durchsicht seiner Schrift [«Pandectarum libri XXI»]. Sollte dessen Zeit nicht dafür ausreichen, möge er wenigstens den letzten Teil zur Theologie [«Partitiones theologicae»] lesen. Die Begutachtung des Übrigen eilt derzeit nicht."

# Labels for entity prediction
labels = ["Location", "Person"] # for v2.1 use capital case for better performance

# Perform entity prediction
print("Predicting entities...", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
entities = model.predict_entities(text, labels, threshold=0.5)

# Display predicted entities and their labels
print("Predicted entities:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
for entity in entities:
    print(entity["text"], "=>", entity["label"])

print("Entity prediction completed", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class NER:
    def __init__(self, model_path="models/checkpoint-6200"):
        self.model = GLiNER.from_pretrained(model_path)

    def predict(self, text, labels=["Location", "Person"], threshold=0.5):
        return self.model.predict_entities(text, labels, threshold)

    def display(self, entities):
        for entity in entities:
            print(entity["text"], "=>", entity["label"])