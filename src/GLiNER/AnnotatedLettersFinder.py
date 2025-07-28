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
    path_to_files = "G:/Meine Ablage/ILU/BullingerDigital/data/letters/*.xml"
    training_folder = "G:/Meine Ablage/ILU/BullingerDigital/src/GLiNER/LettersOriginal"
    files = glob.glob(path_to_files)

    finder = AnnotatedLettersFinder()

    for file in tqdm(files, unit=" Files"):
        source = finder.parse_source(file)

        if source == "TUSTEP":
            finder.move_file_to_training_folder(file, training_folder)

    print("Files parsed:", finder.files_parsed)
    print("Files moved:", finder.files_moved)
    print("Labels found:", finder.source_labels)
