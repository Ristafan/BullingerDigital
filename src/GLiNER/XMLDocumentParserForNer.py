import glob
import random
import xml.etree.ElementTree as ET
from typing import List, Union, Optional, Dict, Any
import os
import re
import json

from tqdm import tqdm


class XMLDocumentParserForNer:
    """
    A class for parsing XML documents with NER annotations.
    Handles both summary and body sections with nested structures.
    """

    def __init__(self, xml_file_path: Optional[str] = None, namespace: Optional[str] = None, xml_string: Optional[str] = None):
        """
        Initialize the parser with either a file path or XML string.

        Args:
            xml_file_path: Path to the XML file to parse
            namespace: XML namespace URI if applicable
            xml_string: XML content as a string
        """
        self.parsed_sentences = []  # List to store all parsed sentences
        self.summary_sentences = []  # List to store summary sentences
        self.body_sentences = []    # List to store body sentences
        self.result = []  # List to store tokenized sentences with NER annotations
        self.cleaned_sentences = []  # List to store cleaned sentences without NER tags
        self.namespace = namespace
        self.root = None

        if xml_file_path:
            self.load_from_file(xml_file_path)
        elif xml_string:
            self.load_from_string(xml_string)

    def load_from_file(self, xml_file_path: str) -> None:
        """
        Load and parse XML from a file.

        Args:
            xml_file_path: Path to the XML file
        """
        if not os.path.exists(xml_file_path):
            raise FileNotFoundError(f"XML file not found: {xml_file_path}")

        try:
            tree = ET.parse(xml_file_path)
            self.root = tree.getroot()
            self._parse_document()
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML file: {e}")

    def load_from_string(self, xml_string: str) -> None:
        """
        Load and parse XML from a string.

        Args:
            xml_string: XML content as string
        """
        try:
            self.root = ET.fromstring(xml_string)
            self._parse_document()
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML string: {e}")

    def _parse_document(self) -> None:
        """
        Parse the entire XML document, extracting summary and body sections.
        """
        if self.root is None:
            raise ValueError("No XML document loaded")

        # Clear previous results
        self.parsed_sentences = []
        self.summary_sentences = []
        self.body_sentences = []

        # Parse summary if it exists
        self._parse_summary()

        # Parse body
        self._parse_body()

        # Combine all sentences in order (summary first, then body)
        self.parsed_sentences = self.summary_sentences + self.body_sentences

    def _parse_summary(self) -> None:
        """
        Parse the summary section if it exists.
        Summary structure: <summary><p>...</p></summary>
        """
        summary_tag = f".//{{{self.namespace}}}summary" if self.namespace else './/summary'
        summary_element = self.root.find(summary_tag)
        if summary_element is not None:
            # Find all <p> elements within the summary
            p_tag = f".//{{{self.namespace}}}p" if self.namespace else './/p'
            p_elements = summary_element.findall(p_tag)

            for p_element in p_elements:
                # Parse the <p> element similar to <s> sections
                sentences = self._parse_paragraph_element(p_element)
                self.summary_sentences.extend(sentences)

    def _parse_body(self) -> None:
        """
        Parse the body section.
        Body structure: <body><div><p><s>...</s></p></div></body>
        """
        body_tag = f".//{{{self.namespace}}}body" if self.namespace else './/body'
        body_element = self.root.find(body_tag)
        if body_element is None:
            raise ValueError("No <body> element found in XML document")

        # Find all <div> elements within the body
        div_tag = f".//{{{self.namespace}}}div" if self.namespace else './/div'
        div_elements = body_element.findall(div_tag)
        if not div_elements:
            raise ValueError("No <div> elements found in <body>")

        for div_element in div_elements:
            # Find all <p> elements within each div
            p_tag = f".//{{{self.namespace}}}p" if self.namespace else './/p'
            p_elements = div_element.findall(p_tag)
            if not p_elements:
                continue  # Skip divs without paragraphs

            for p_element in p_elements:
                # Find all <s> elements within each paragraph
                s_tag = f".//{{{self.namespace}}}s" if self.namespace else './/s'
                s_elements = p_element.findall(s_tag)
                if not s_elements:
                    continue  # Skip paragraphs without sentences

                for s_element in s_elements:
                    sentences = self.parse_section(s_element)
                    self.body_sentences.extend(sentences)

    def _parse_paragraph_element(self, p_element: ET.Element) -> List[str]:
        """
        Parse a paragraph element, treating it similarly to a section.
        This is used for summary paragraphs.

        Args:
            p_element: XML paragraph element

        Returns:
            List of parsed sentences
        """
        return self.parse_section(p_element)

    def parse_section(self, section_element: ET.Element) -> List[str]:
        """
        Parse an XML section element and extract text with NER annotations.

        Args:
            section_element: XML Element representing a section (like <s> or <p>)

        Returns:
            List of strings where:
            - First string is the main sentence with NER annotations
            - Additional strings are note sentences with NER annotations
        """

        def process_element(element: ET.Element, in_note: bool = False) -> tuple[str, List[str]]:
            """
            Recursively process an XML element and its children.

            Args:
                element: Current XML element to process
                in_note: Whether we're currently inside a <note> element

            Returns:
                tuple: (text_for_current_context, list_of_note_texts)
            """
            result_text = ""
            note_texts = []

            # Add text before any child elements
            if element.text:
                result_text += element.text

            # Process child elements
            for child in element:
                if child.tag == 'note' or (self.namespace and child.tag == f"{{{self.namespace}}}note"):
                    # Handle note elements - they become separate sentences
                    note_content, nested_notes = process_element(child, in_note=True)
                    note_texts.append(note_content)
                    note_texts.extend(nested_notes)

                elif child.tag == 'placeName' or (self.namespace and child.tag == f"{{{self.namespace}}}placeName"):
                    # Handle place name annotations
                    place_text = self._get_element_text(child)
                    if place_text:
                        annotated_text = self._create_ner_annotation(place_text, 'Place')
                        result_text += annotated_text

                elif child.tag == 'persName' or (self.namespace and child.tag == f"{{{self.namespace}}}persName"):
                    # Handle person name annotations
                    person_text = self._get_element_text(child)
                    if person_text:
                        annotated_text = self._create_ner_annotation(person_text, 'Pers')
                        result_text += annotated_text

                else:
                    # Handle other elements (like <foreign>) - just extract text content
                    child_text, child_notes = process_element(child, in_note)
                    result_text += child_text
                    note_texts.extend(child_notes)

                # Add text that comes after this child element
                if child.tail:
                    result_text += child.tail

            return result_text, note_texts

        # Process the main section element
        main_text, notes = process_element(section_element)

        # Clean up whitespace
        main_text = ' '.join(main_text.split())
        notes = [' '.join(note.split()) for note in notes if note.strip()]

        # Return results - main text first, then notes
        result = [main_text] if main_text.strip() else []
        result.extend(notes)

        return result if result else [""]

    def _get_element_text(self, element: ET.Element) -> str:
        """
        Get all text content from an element, including nested elements.
        This handles cases where placeName/persName might contain other elements.
        """
        text_parts = []

        if element.text:
            text_parts.append(element.text)

        for child in element:
            # For placeName and persName, we want to include text from nested elements
            child_text = self._get_element_text(child)
            if child_text:
                text_parts.append(child_text)
            if child.tail:
                text_parts.append(child.tail)

        return ''.join(text_parts)

    def _create_ner_annotation(self, text: str, entity_type: str) -> str:
        """
        Create NER annotation for a given text and entity type.

        Args:
            text: The text to annotate
            entity_type: Either 'Place' or 'Pers'

        Returns:
            Annotated text string
        """
        words = text.split()
        if len(words) == 1:
            return f"single{entity_type}{words[0]}"
        else:
            annotated_text = f"start{entity_type}{words[0]} {' '.join(words[1:-1])} end{entity_type}{words[-1]}" if len(words) > 1 else f"start{entity_type}{words[0]} end{entity_type}{words[0]}"
            # Clean up extra spaces
            return ' '.join(annotated_text.split())

    def get_all_sentences(self) -> List[str]:
        """
        Get all parsed sentences from the document.

        Returns:
            List of all parsed sentences (summary + body)
        """
        return self.parsed_sentences.copy()

    def get_summary_sentences(self) -> List[str]:
        """
        Get only the summary sentences.

        Returns:
            List of summary sentences
        """
        return self.summary_sentences.copy()

    def get_body_sentences(self) -> List[str]:
        """
        Get only the body sentences.

        Returns:
            List of body sentences
        """
        return self.body_sentences.copy()

    def print_parsed_content(self) -> None:
        """
        Print all parsed content in a formatted way.
        """
        print("=" * 60)
        print("PARSED XML DOCUMENT CONTENT")
        print("=" * 60)

        if self.summary_sentences:
            print("\nSUMMARY SENTENCES:")
            print("-" * 30)
            for i, sentence in enumerate(self.summary_sentences, 1):
                print(f"{i:2d}. {sentence}")

        if self.body_sentences:
            print("\nBODY SENTENCES:")
            print("-" * 30)
            for i, sentence in enumerate(self.body_sentences, 1):
                print(f"{i:2d}. {sentence}")

        print(f"\nTOTAL SENTENCES: {len(self.parsed_sentences)}")

    def tokenize_and_extract_ner(self) -> List[Dict[str, Any]]:
        """
        Tokenize all parsed sentences and extract NER information.

        Returns:
            List of dictionaries with tokenized text and NER annotations
        """
        self.result = []

        for sentence in self.parsed_sentences:
            tokenized_data = self._process_sentence_for_ner(sentence)
            self.result.append(tokenized_data)

        return self.result

    def _process_sentence_for_ner(self, sentence: str) -> Dict[str, Any]:
        """
        Process a single sentence to extract tokens and NER information.

        Args:
            sentence: Input sentence with NER tags

        Returns:
            Dictionary with tokenized_text and ner lists
        """
        # First tokenize the sentence as-is (with NER tags)
        tokens = re.findall(r'\w+(?:[-_]\w+)*|\S', sentence)

        # Extract NER information from the tokens
        ner_annotations = []
        clean_tokens = []
        token_idx = 0

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Check if this token is a NER tag
            if token.startswith(('startPers', 'endPers', 'singlePers', 'startPlace', 'endPlace', 'singlePlace')):
                # Extract the tag type and word
                if token.startswith('startPers'):
                    tag_type = 'startPers'
                    word = token[9:]  # Remove 'startPers'
                    entity_type = 'person'
                elif token.startswith('endPers'):
                    tag_type = 'endPers'
                    word = token[7:]  # Remove 'endPers'
                    entity_type = 'person'
                elif token.startswith('singlePers'):
                    tag_type = 'singlePers'
                    word = token[10:]  # Remove 'singlePers'
                    entity_type = 'person'
                elif token.startswith('startPlace'):
                    tag_type = 'startPlace'
                    word = token[10:]  # Remove 'startPlace'
                    entity_type = 'location'
                elif token.startswith('endPlace'):
                    tag_type = 'endPlace'
                    word = token[8:]  # Remove 'endPlace'
                    entity_type = 'location'
                elif token.startswith('singlePlace'):
                    tag_type = 'singlePlace'
                    word = token[11:]  # Remove 'singlePlace'
                    entity_type = 'location'

                # Add the word to clean tokens
                clean_tokens.append(word)

                # Handle different tag types
                if tag_type.startswith('single'):
                    # Single word entity
                    ner_annotations.append([token_idx, token_idx, entity_type])
                elif tag_type.startswith('start'):
                    # Find the corresponding end tag
                    start_pos = token_idx
                    end_pos = start_pos

                    # Look ahead for the end tag
                    j = i + 1
                    while j < len(tokens):
                        next_token = tokens[j]
                        expected_end = f"end{'Pers' if 'Pers' in tag_type else 'Place'}"

                        if next_token.startswith(expected_end):
                            # Found the end tag, extract its word
                            end_word = next_token[len(expected_end):]
                            clean_tokens.append(end_word)
                            end_pos = token_idx
                            i = j  # Skip to after the end tag
                            break
                        else:
                            # Regular token between start and end
                            clean_tokens.append(next_token)
                            token_idx += 1
                        j += 1

                    ner_annotations.append([start_pos, end_pos + 1, entity_type])

                token_idx += 1
            else:
                # Regular token, just add it
                clean_tokens.append(token)
                token_idx += 1

            i += 1

        # Recreate original sentence string without NER tags
        stripped_tokens = [token for token in clean_tokens if token.strip()]
        combined_tokens = ' '.join(clean_tokens)
        self.cleaned_sentences.append(combined_tokens)


        return {
            "tokenized_text": clean_tokens,
            "ner": ner_annotations
        }

    def save_tokenized_data_as_json(self, output_file_path: str) -> None:
        """
        Save the tokenized data with NER annotations as JSON.

        Args:
            output_file_path: Path where to save the JSON file
        """
        tokenized_data = self.tokenize_and_extract_ner()

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(tokenized_data, f, ensure_ascii=False, indent=2)
            print(f"Tokenized data saved to: {output_file_path}")
        except Exception as e:
            raise IOError(f"Failed to save JSON file: {e}")


# Example usage and test cases
if __name__ == "__main__":
    namespace = TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
    xml_file_path = "G:/Meine Ablage/ILU/BullingerDigital/src/GLiNER/LettersOriginal/1.xml"
    train_data = []
    skipped_files_counter = 0
    skipped_files = []

    new_path = "G:/Meine Ablage/ILU/NEW"
    all_files = glob.glob(new_path + "/*.xml")
    all_files_basename = [os.path.basename(file) for file in all_files]
    num_relevant_sentences = 0
    num_not_relevant_sentences = 0
    training_sentences_and_annotations = {}
    # skipped_sentences_and_annotations = {}  Not used in this version

    rand = random.randint(0, 1)

    for file in tqdm(glob.glob("G:/Meine Ablage/ILU/BullingerDigital/src/GLiNER/LettersOriginal/*.xml")):
        basename = os.path.basename(file)

        if basename in all_files_basename:
            skipped_files.append(basename)
            continue
        #if rand == 1 and skipped_files_counter < 300:
        #    skipped_files_counter += 1
        #    skipped_files.append(basename)
        #    continue

        parser = XMLDocumentParserForNer(file, namespace)
        # parser.print_parsed_content()
        tokenized_data = parser.tokenize_and_extract_ner()

        # Store the sentences and their annotations in a dictionary with numbering of the tokens
        for idx, data in enumerate(tokenized_data):
            training_sentences_and_annotations[f"{basename}_{idx}"] = {
                "tokenized_text": [str(index) + "_" + token for index, token in enumerate(data["tokenized_text"])],  # Numbering of tokens for better readability
                "ner": data["ner"],
                "original_sentence": parser.cleaned_sentences[idx]
            }

        # Count number of sentences in the document
        num_sentences = len(tokenized_data)

        # Remove empty sentences and sentences that don't contain any NER annotations
        tokenized_data = [data for data in tokenized_data if data["tokenized_text"] and data["ner"]]
        train_data.extend(tokenized_data)

        # Count remaining sentences
        num_remaining_sentences = len(tokenized_data)
        num_relevant_sentences += num_remaining_sentences
        num_not_relevant_sentences += (num_sentences - num_remaining_sentences)

    print(f"Skipped files list: {skipped_files}")
    print(f"Skipped files: {skipped_files_counter}")
    print(f"Total relevant sentences: {num_relevant_sentences}")
    print(f"Total not relevant sentences: {num_not_relevant_sentences}")

    # Save the tokenized data to a JSON file
    with open("training1.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # Save the training sentences and annotations to a JSON file
    with open("training_sentences_and_annotations(1).json", 'w', encoding='utf-8') as f:
        json.dump(training_sentences_and_annotations, f, ensure_ascii=False, indent=2)