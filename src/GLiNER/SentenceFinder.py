import json
from pprint import pprint


def get_sentence_finder(filename_and_sentence_index):
    with open("predicted_entities.json", "r", encoding="utf-8") as f:
        sentences = json.load(f)

    for i in sentences:
        sentence_index = i["sentence_pair"]
        if sentence_index == filename_and_sentence_index:

            pprint(f"Found sentence: {i["sentence_pair"]}")
            print()

            pprint(f"Original sentence: {i['original_sentence']}")
            print()

            print(f"Predicted entities:")
            for entity in i["entities"]:
                token_start, token_end = mapping_to_token_indices(entity, i["original_sentence"])

                if token_start is not None and token_end is not None:
                    print(f"[{token_start}, {token_end}, {entity['label']}] --> {entity['text']} score: {round(entity['score'], 2)}\n")
                else:
                    print(f"[?, ?, {entity['label']}] --> {entity['text']} score: {round(entity['score'], 2)} (Token index mapping failed)\n")

            print(f"Labelled entities:")
            for label in i["labelled_entities"]:
                print(f"{label}")
            break

    else:
        print(f"Sentence with index {filename_and_sentence_index} not found.")


def mapping_to_token_indices(entity, original_sentence):
    start_char = entity['start']
    end_char = entity['end']

    # Map character positions to token indices
    token_start, token_end = None, None
    for i, token in enumerate(original_sentence.split(" ")):

        # Get start and end positions of token in original sentence
        token_char_start = original_sentence.find(token)
        token_char_end = token_char_start + len(token)

        if token_char_start <= start_char < token_char_end:
            token_start = i
        if token_char_start < end_char <= token_char_end:
            token_end = i

    return token_start, token_end


get_sentence_finder("2036.xml_1")
