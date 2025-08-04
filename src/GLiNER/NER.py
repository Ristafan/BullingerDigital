import os
from datetime import datetime
import torch
import json
from tqdm import tqdm

print("Set environment variables for offline mode", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("Load GLiNER Entity Prediction", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
from gliner.model import GLiNER

""" Test code for GLiNER entity prediction
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
"""


class NER:
    def __init__(self, model_path="models/checkpoint-6200"):
        self.model = GLiNER.from_pretrained(model_path)

    def predict(self, text, labels=["Location", "Person"], threshold=0.5):
        return self.model.predict_entities(text, labels, threshold)

    def display(self, entities):
        for entity in entities:
            print(entity["text"], "=>", entity["label"])


if __name__ == "__main__":
    with open("training_sentences_and_annotations(1).json", "r", encoding="utf-8") as f:
        data = json.load(f)

    ner = NER()

    for sentence in tqdm(data.keys()):
        with open("predicted_entities.txt", "a", encoding="utf-8") as f:

            f.write(f"Processing sentence: {sentence}\n")
            original_sentence = data[sentence]["original_sentence"]
            entities = ner.predict(original_sentence, labels=["Location", "Person"], threshold=0.5)

            f.write(f"Original Sentence: {" ".join(data[sentence]['tokenized_text'])}\n")
            f.write("Labelled Entities:\n")
            f.write(str(data[sentence]["ner"]))
            f.write("\nPredicted Entities:\n")
            for entity in entities:
                start_char = entity['start']
                end_char = entity['end']

                # Map character positions to token indices
                token_start, token_end = None, None
                for i, token in enumerate(data[sentence]['tokenized_text']):
                    token_index, token_text = token.split("_", 1)
                    token_index = int(token_index)

                    # Get start and end positions of token in original sentence
                    token_char_start = original_sentence.find(token_text)
                    token_char_end = token_char_start + len(token_text)

                    if token_char_start <= start_char < token_char_end:
                        token_start = token_index
                    if token_char_start < end_char <= token_char_end:
                        token_end = token_index

                if token_start is not None and token_end is not None:
                    f.write(f"[{token_start}, {token_end}, {entity['label']}] --> {entity['text']} score: {round(entity['score'], 2)}\n")
                else:
                    f.write(f"[?, ?, {entity['label']}] --> {entity['text']} score: {round(entity['score'], 2)} (Token index mapping failed)\n")

            f.write("\n")
            f.write("=" * 50 + "\n")
            f.write("\n")
