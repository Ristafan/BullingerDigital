# Bullinger-Digital-Merging
This project aims to create a sustainable foundation for automated entity recognition, enhancing the discoverability and integration of the Bullinger correspondence into broader knowledge systems.


## Meta information about XML files
The class XmlParser inside `src/PersonPlaceParser/XmlParser.py` is responsible for parsing the XML files and extracting information about persons and places. It counts the occurrences of `persName` and `placeName` elements in both the summary and body sections of each file.

#### Usage
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