# Bullinger-Digital-Merging
This project aims to create a sustainable foundation for automated entity recognition, enhancing the discoverability and integration of the Bullinger correspondence into broader knowledge systems.

## Meta information about XML files
The class XmlParser inside `src/PersonPlaceParser/XmlParser.py` is responsible for parsing the XML files and extracting information about persons and places. It counts the occurrences of `persName` and `placeName` elements in both the summary and body sections of each file.

### Usage
To use the `XmlParser`, at the bottom of the `XmlParser.py` file, you can find an example of how to instantiate the class and call the `parse_xml_files` method. Specify the directory containing the XML files you want to parse.

### Example Json Format of persName and placeName counts per file

```json
{
  "Filename": {
    "summary": {
      "persNames": {
        "": {
          "ref": "",
          "count": 0
        }
      },
      "placeNames": {
        "": {
          "ref": "",
          "count": 0
        }
      }
    },
    "body": {
      "persNames": {
        "": {
          "ref": "",
          "count": 0
        }
      },
      "placeNames": {
        "": {
          "ref": "",
          "count": 0
        }
      }
    }
  }
}
```

## Fine-tuning the model
To fine-tune the model, you first need to extract training data from the XML files using the `XMLDocumentParserForNER.py`. Then you need to train the model using the `FineTuner.py` script. Both files can be found in the `src/GLiNER` directory.

### Extracting Training Data
The XMLDocumentParserForNer class is designed to parse XML documents containing Named Entity Recognition (NER) annotations. It processes both the summary and body sections of the XML, extracts sentences, and identifies entities such as persons and places. The class also supports handling nested structures and reference IDs for entities.

### Usage
- To use the `XMLDocumentParserForNER`, you can find an example at the bottom of the `XMLDocumentParserForNER.py` file. You need to specify the directory containing the XML files. Two output files will be generated: `training.json` and `training_readable.json`. The `training.json` file is used for training the model, while the `training_readable.json` file is a more human-readable format and will be used by the `NER` class to make predictions using the fine-tuned model.
- To use the `FineTuner`, you can find an example at the bottom of the `FineTuner.py` file. You need to specify the path to the training dataset in JSON format (that we generated in the previous step) and the name of the pre-trained model you want to fine-tune.

### Json Format of Training Data
Following is an example of the JSON format used for training data. The `tokenized_text` field contains the tokenized version of the text, and the `ner` field contains the named entity recognition annotations, where each entity is represented by a start index, end index, and label.
```json
[
  {
    "tokenized_text": [
      "De",
      "rebus",
      "aliis",
      ",",
      "quę",
      "scire",
      "vos",
      "refert",
      ",",
      "copiose",
      ",",
      "credo",
      ",",
      "vos",
      "Hedio",
      "noster",
      "certiores",
      "facit",
      "."
    ],
    "ner": [
      [
        14,
        14,
        "person"
      ]
    ]
  },
...
]
```

## Using the Fine-tuned Model
To use the fine-tuned model for Named Entity Recognition (NER), you can utilize the `NER` class found in the `src/GLiNER` directory. This class is designed to load the fine-tuned model and make predictions on new text data.

### Usage
To use the `NER` class, you can find an example at the bottom of the `NER.py` file. You need to specify the path to the fine-tuned model and the text you want to analyze for named entities and the variable `labels_available` if true labels for the entities are available from the test set. In the line `entities = ner.predict(original_sentence, labels=["Location", "Person"], threshold=0.5)`, specify which labels shou be predicted. Right after this line are two method calls, where you can specify own additional rules to filter out entities that you do not want to be included in the final output, or entities that you want to be included in the final output.
The script will return two files: `predicted_entities.json` and `predicted_entities.txt`, where `predicted_entities.json` contains the predicted entities in JSON format, and `predicted_entities.txt` contains the predicted entities in a human-readable format.

### Analyzing the Results
To analyze the results of the NER predictions, you can use the `ResultComparer` class found in the `src/GLiNER` directory. This class is designed to evaluate the performance of the NER model by comparing the predicted entities with the ground truth labels.
At the bottom of the `ResultComparer.py` file, you can find an example of how to use the class. You need to specify the paths to the json file created by the `NER.py` script. The script will output various evaluation metrics such as precision, recall, and F1-score, which can help you assess the performance of your NER model. It will also output a file `results.html` that contains color-coded results of each sentence, highlighting the entities that were correctly predicted, missed, or incorrectly predicted.

## Additional Scripts
In the folder `src/EntityLinking` you can find additional scripts that can be used to evaluate ambiguities in the id labels for persons and places. The `PlaceExtractor.py` and `PersonExtractor.py` scripts are designed to extract the entities from the XML files, from https://github.com/bullinger-digital/bullinger-korpus-tei/tree/main/data/index.
Based on the output json files from these scripts, the files `PlaceAmbiguityAnalyzer.py` and `PersonAmbiguityAnalyzer.py` can be used to analyze the ambiguities in the id labels for persons and places. These scripts will print some information about the ambiguities, such as the number of entities with the same string but different id labels and other metrics that can help you understand the quality of the data.

## Requirements
To run the scripts in this project, you need to have Python 3.8 or higher installed. You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Background on project
The Bullinger-Digital-Merging project is part of the Bullinger Digital initiative, which aims to digitize and make accessible the correspondence of the theologian Johann Heinrich Bullinger. The project focuses on enhancing the discoverability and integration of the Bullinger correspondence into broader knowledge systems through automated entity recognition and linking.
For more information about the Bullinger Digital initiative, you can visit the [Bullinger Digital website](https://www.bullinger-digital.ch/).

## 