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

    def mapping_to_token_indices(self, entity, data):
        start_char = entity['start']
        end_char = entity['end']

        # Map character positions to token indices
        token_start, token_end = None, None
        for i, token in enumerate(data[sentence_pair]['tokenized_text']):
            token_index, token_text = token.split("_", 1)
            token_index = int(token_index)

            # Get start and end positions of token in original sentence
            token_char_start = original_sentence.find(token_text)
            token_char_end = token_char_start + len(token_text)

            if token_char_start <= start_char < token_char_end:
                token_start = token_index
            if token_char_start < end_char <= token_char_end:
                token_end = token_index

        return token_start, token_end

    @staticmethod
    def save_to_txt(sentence_pair, data, entities):
        with open("predicted_entities.txt", "a", encoding="utf-8") as f:

            f.write(f"Processing sentence: {sentence_pair}\n")

            f.write(f"Original Sentence: {" ".join(data[sentence_pair]['tokenized_text'])}\n")
            f.write("Labelled Entities:\n")
            f.write(str(data[sentence_pair]["ner"]))
            f.write("\nPredicted Entities:\n")

            for entity in entities:
                token_start, token_end = ner.mapping_to_token_indices(entity, data)

                if token_start is not None and token_end is not None:
                    f.write(f"[{token_start}, {token_end}, {entity['label']}] --> {entity['text']} score: {round(entity['score'], 2)}\n")
                else:
                    f.write(f"[?, ?, {entity['label']}] --> {entity['text']} score: {round(entity['score'], 2)} (Token index mapping failed)\n")

            f.write("\n")
            f.write("=" * 50 + "\n")
            f.write("\n")

    @staticmethod
    def save_to_json(sentence_pair, original_sentence, entities):
        output = {
            "sentence_pair": sentence_pair,
            "original_sentence": original_sentence,
            "labelled_entities": data[sentence_pair]["ner"],
            "entities": entities
        }

        return output


if __name__ == "__main__":
    all_predicted_entities = []

    # Load the data from the JSON file
    with open("training_sentences_and_annotations(1).json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize the NER model
    ner = NER()
    i = 0

    # Process each sentence pair in the data
    for sentence_pair in tqdm(data.keys()):
        # Predict entities for each sentence pair
        original_sentence = data[sentence_pair]["original_sentence"]
        entities = ner.predict(original_sentence, labels=["Location", "Person"], threshold=0.5)

        # Save the results to a text file
        ner.save_to_txt(sentence_pair, data, entities)

        # Save the results to later use in a JSON file
        all_predicted_entities.append(ner.save_to_json(sentence_pair, original_sentence, entities))

        # Save the results to later us in an Excel file



        i += 1
        if i == 20:
            break

    # Save all predicted entities to a JSON file
    with open("predicted_entities.json", "a", encoding="utf-8") as f:
        json.dump(all_predicted_entities, f, ensure_ascii=False, indent=4)


