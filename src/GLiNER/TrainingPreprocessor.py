import glob
import re
import xml.etree.ElementTree as ET


class TrainingPreprocessor:
    def __init__(self):
        self.entities = ["Person", "Location"]
        self.content = []

    def parse_whole_sentences(self, filepath):
        # create element tree object
        root = self.create_element_tree_object(filepath)
        TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"

        # Iterate through the XML summary to get person and location names
        summary_tag = f"{{{TEI_NAMESPACE}}}summary"
        summary_element = root.find(f".//{summary_tag}")

        if summary_element is not None:
            self.extract_sentences_and_entities_body(summary_element, [])
        else:
            print("No summary found in the XML file.")

        # Iterate through the XML body to get person and location names
        body_tag = f"{{{TEI_NAMESPACE}}}body"
        body_element = root.find(f".//{body_tag}")

        if body_element is not None:
            self.extract_sentences_and_entities_body(body_element, [])
        else:
            print("No body found in the XML file.")

    def extract_sentences_and_entities_body(self, element, ancestor_path):
        current_path = ancestor_path + [element]
        for child in element:

            #print(child.attrib)
            #if "{http://www.w3.org/XML/1998/namespace}id" in child.attrib:
            #    print(f"Processing element with xml:id: {child.attrib["{http://www.w3.org/XML/1998/namespace}id"]}")

            # Check <s> tags for sentences
            if child.tag[29:] == "s":
                self.parse_paragraph(child, ancestor_path)

            elif self.has_children(child):
                self.extract_sentences_and_entities_body(child, current_path)

    def parse_paragraph(self, paragraph, ancestor_path):
        # Extract all Entities in the paragraph
        paragraph_entities = {"Person": {}, "Location": {}}

        # Extract only the direct text from the paragraph excluding children content except entities
        text_parts = []

        # Add the element's direct text (before first child)
        if paragraph.text:
            text_parts.append(paragraph.text)

        # Add tail text from each child (text that comes after each child element)
        for child in paragraph:
            if child.tag.endswith("persName") or child.tag.endswith("placeName"):
                text_parts.append(child.text or "")

            if child.tail:
                text_parts.append(child.tail)

        paragraph_text = ''.join(text_parts).strip()

        for element in paragraph:
            # Check if there are footnote children
            if element.tag.endswith("note"):
                self.parse_paragraph(element, ancestor_path)

            # Check if the element is a person or place name
            if element.tag.endswith("persName"):
                # Get the text content of persName element
                person_text = element.text or ""
                if person_text.strip():
                    paragraph_entities["Person"][person_text] = person_text
            elif element.tag.endswith("placeName"):
                # Get the text content of placeName element
                place_text = element.text or ""
                if place_text.strip():
                    paragraph_entities["Location"] = place_text

        print(f"Extracted paragraph: {paragraph_text}")
        self.content.append([paragraph_text, paragraph_entities])

    @staticmethod
    def get_direct_text_only(element):
        """
        Extract only the direct text content of an element, excluding child element text,
        but including text that is separated by child elements (tail text).
        """


    @staticmethod
    def create_element_tree_object(filepath):
        ET.XMLParser(encoding="utf-8")
        tree = ET.parse(filepath)
        return tree.getroot()

    @staticmethod
    def has_children(element):
        """Check if an XML element has children."""
        return any(True for _ in element)

    @staticmethod
    def is_inside_footnote_with_path(ancestor_path):
        """Check if any ancestor in the path is a footnote."""
        for ancestor in ancestor_path:
            if ancestor.get("type") == "footnote":
                return True

        return False


if __name__ == "__main__":
    files_folder = "G:/Meine Ablage/ILU/BullingerDigital/src/GLiNER/LettersOriginal/*.xml"
    files = glob.glob(files_folder)

    print(files[0])

    tp = TrainingPreprocessor()
    tp.parse_whole_sentences(files[0])

    for content in tp.content:
        print(content[0])
        print(content[1])
        print()


    # S. Rogo, clarissime d. Bullingere, ut scriptum
    # hoc meum, quando vacabit, perlegere digneris, et si minus est ocii,
    # saltem posteriorem partem
    # legas, quę est de disciplina ecclesię.