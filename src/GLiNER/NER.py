import os
from datetime import datetime
import pandas
import torch
import json
from tqdm import tqdm

print("Set environment variables for offline mode", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("Load GLiNER Entity Prediction", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
from gliner.model import GLiNER


class NER:
    def __init__(self, model_path="models_01/checkpoint-6200"):
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

    def remove_irrelevant_entities(self, entities):
        for entity in entities:
            # Remove entities that are lowercase
            if entity['text'].islower():
                print(f"Entity removed: {entity['text']}")
                entities.remove(entity)
                continue
            # Remove entities that are single or two characters long
            if len(entity['text']) <= 2:
                print(f"Entity removed: {entity['text']}")
                entities.remove(entity)
                continue

            # Remove specific irrelevant entities
            if entity['text'] in ["", " ", "?", "!", ":", ";", ",", ".", "-", "_"]:
                print(f"Entity removed: {entity['text']}")
                entities.remove(entity)
                continue
            if entity['text'].lower() == "vale":
                print(f"Entity removed: {entity['text']}")
                entities.remove(entity)
                continue
            if entity['text'].lower() == "rande":
                print(f"Entity removed: {entity['text']}")
                entities.remove(entity)
                continue

        return entities

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
    def convert_to_json_format(sentence_pair, original_sentence, entities):
        output = {
            "sentence_pair": sentence_pair,
            "original_sentence": original_sentence,
            "labelled_entities": data[sentence_pair]["ner"],
            "entities": entities
        }

        return output

    @staticmethod
    def save_meta_information_to_excel(all_predicted_entities):
        # Extract meta information from the predicted entities
        filename_and_sentence_index = []
        num_predictions = []
        num_labels = []

        num_person_predictions = []
        num_location_predictions = []
        num_person_labels = []
        num_location_labels = []

        for sentence in all_predicted_entities:
            filename_and_sentence_index.append(sentence["sentence_pair"])
            num_predictions.append(len(sentence["entities"]))
            num_labels.append(len(sentence["labelled_entities"]))

            num_person_predictions.append(len([e for e in sentence["entities"] if e["label"] == "Person"]))
            num_person_labels.append(len([e for e in sentence["labelled_entities"] if e[2] == "person"]))
            num_location_predictions.append(len([e for e in sentence["entities"] if e["label"] == "Location"]))
            num_location_labels.append(len([e for e in sentence["labelled_entities"] if e[2] == "location"]))

        # Create a DataFrame and save it to an Excel file
        df = pandas.DataFrame({
            "filename_and_sentence_index": filename_and_sentence_index,
            "num_predictions": num_predictions,
            "num_labels": num_labels,
            "num_person_predictions": num_person_predictions,
            "num_person_labels": num_person_labels,
            "num_location_predictions": num_location_predictions,
            "num_location_labels": num_location_labels
        })

        df.to_excel("meta_information.xlsx", index=False, engine='openpyxl')


if __name__ == "__main__":
    all_predicted_entities = []

    # Load the data from the JSON file
    with open("test_final_sentences_and_annotations.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize the NER model
    ner = NER()

    # Process each sentence pair in the data
    for sentence_pair in tqdm(data.keys()):
        # Predict entities for each sentence pair
        original_sentence = data[sentence_pair]["original_sentence"]
        entities = ner.predict(original_sentence, labels=["Location", "Person"], threshold=0.5)

        cleaned_entities = ner.remove_irrelevant_entities(entities)

        # Save the results to a text file
        #ner.save_to_txt(sentence_pair, data, cleaned_entities)

        # Save the results to later use in a JSON file
        all_predicted_entities.append(ner.convert_to_json_format(sentence_pair, original_sentence, entities))

    # Save all predicted entities to a JSON file
    print("Saving predicted entities to JSON file", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with open("predicted_entities_final_removed_unnecessary.json", "a", encoding="utf-8") as f:
        json.dump(all_predicted_entities, f, ensure_ascii=False, indent=4)

    # Save meta information to an Excel file
    #print("Saving meta information to Excel file", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    #ner.save_meta_information_to_excel(all_predicted_entities)


