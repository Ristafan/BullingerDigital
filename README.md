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
- To use the `XMLDocumentParserForNER`, you can find an example at the bottom of the `XMLDocumentParserForNER.py` file. You need to specify the directory containing the XML files. Two output files will be generated: `training.json` and `training_readable.json`. The `training.json` file is used for training the model, while the `training_readable.json` file is a more human-readable format.
- To use the `FineTuner`, you can find an example at the bottom of the `FineTuner.py` file. You need to specify the path to the training dataset in JSON format (that we generated in the previous step) and the name of the pre-trained model you want to fine-tune.

