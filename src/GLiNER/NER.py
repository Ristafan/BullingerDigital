from datetime import datetime
import json
from tqdm import tqdm
from gliner.model import GLiNER


class NER:
    def __init__(self, model_path, labels_available):
        self.model = GLiNER.from_pretrained(model_path)
        self.labels_available = labels_available

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
    def remove_irrelevant_entities(entities):
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
    def add_relevant_entities(entities, original_sentence):
        # This method is a placeholder for adding relevant entities
        pass

    def save_to_txt(self, sentence_pair, data, entities):
        with open("predicted_entities.txt", "a", encoding="utf-8") as f:

            f.write(f"Processing sentence: {sentence_pair}\n")

            f.write(f"Original Sentence: {" ".join(data[sentence_pair]['tokenized_text'])}\n")

            if self.labels_available:
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

    def convert_to_json_format(self, sentence_pair, original_sentence, entities):
        if not self.labels_available:
            data[sentence_pair]["ner"] = []

        output = {
            "sentence_pair": sentence_pair,
            "original_sentence": original_sentence,
            "labelled_entities": data[sentence_pair]["ner"],
            "entities": entities
        }

        return output


if __name__ == "__main__":
    model_path = ""  # Replace with your actual model path
    labels_available = True  # Set to True if labels are available in the data and false if not

    all_predicted_entities = []

    # Load the data from the JSON file
    with open("training_readable.json", "r", encoding="utf-8") as f:  # Replace with your actual file path
        data = json.load(f)

    # Initialize the NER model
    ner = NER(model_path, labels_available)

    # Process each sentence pair in the data
    for sentence_pair in tqdm(data.keys()):
        # Predict entities for each sentence pair
        original_sentence = data[sentence_pair]["original_sentence"]
        entities = ner.predict(original_sentence, labels=["Location", "Person"], threshold=0.5)

        # Add and remove specific entities using own methods
        cleaned_entities = ner.remove_irrelevant_entities(entities)
        # cleaned_entities = ner.add_relevant_entities(entities, original_sentence)

        # Save the results to a text file
        ner.save_to_txt(sentence_pair, data, cleaned_entities)

        # Save the results to later use in a JSON file
        all_predicted_entities.append(ner.convert_to_json_format(sentence_pair, original_sentence, entities))

    # Save all predicted entities to a JSON file
    print("Saving predicted entities to JSON file", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with open("predicted_entities.json", "a", encoding="utf-8") as f:
        json.dump(all_predicted_entities, f, ensure_ascii=False, indent=4)
