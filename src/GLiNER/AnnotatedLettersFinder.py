import glob
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm


class AnnotatedLettersFinder:
    def __init__(self):
        self.files_parsed = 0
        self.files_moved = 0
        self.source_labels = {}

    def parse_source(self, filepath):
        self.files_parsed += 1

        ET.XMLParser(encoding="utf-8")
        tree = ET.parse(filepath)
        root = tree.getroot()
        source = root.get("source")

        if source not in self.source_labels.keys():
            self.source_labels[source] = 1
        else:
            self.source_labels[source] += 1

        return source

    def move_file_to_training_folder(self, file, destination):
        self.files_moved += 1
        shutil.copy2(file, destination)


if __name__ == "__main__":
    path_to_files = "C:/Users/MartinFaehnrich/Documents/BullingerDigi/data/letters/*.xml"
    training_folder = "C:/Users/MartinFaehnrich/Documents/BullingerDigi/src/GLiNER/LettersOriginal"
    files = glob.glob(path_to_files)

    finder = AnnotatedLettersFinder()

    for file in tqdm(files, unit=" Files"):
        try:
            source = finder.parse_source(file)
        except ET.ParseError:
            print(f"Error parsing file: {file}")
            continue

        if source == "keine":
            continue
        elif source == "HTR":
            continue
        else:
            finder.move_file_to_training_folder(file, training_folder)

    print("Files parsed:", finder.files_parsed)
    print("Files moved:", finder.files_moved)
    print("Labels found:", finder.source_labels)

# If the directory needs to be cleaned up, rin the following command in the Windows terminal:
# rmdir /S /Q "path_to_directory"
