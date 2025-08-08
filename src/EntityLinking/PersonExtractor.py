import xml.etree.ElementTree as ET
import json
from typing import Dict, List, Tuple


class TEIPersonExtractor:
    """
    A class to extract person information from TEI XML files and convert to structured JSON.

    The class parses XML files containing person data in TEI format and creates a dictionary
    where each person's xml:id maps to lists of surnames and forenames (including aliases).
    """

    def __init__(self):
        self.namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
        self.persons_dict = {}

    def extract_persons_from_file(self, xml_file_path: str) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Extract person data from an XML file.

        Args:
            xml_file_path (str): Path to the XML file

        Returns:
            Dict[str, Tuple[List[str], List[str]]]: Dictionary with person IDs as keys
            and tuples of (surnames, forenames) lists as values
        """
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        return self._extract_persons_from_root(root)

    def extract_persons_from_string(self, xml_string: str) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Extract person data from an XML string.

        Args:
            xml_string (str): XML content as string

        Returns:
            Dict[str, Tuple[List[str], List[str]]]: Dictionary with person IDs as keys
            and tuples of (surnames, forenames) lists as values
        """
        root = ET.fromstring(xml_string)
        return self._extract_persons_from_root(root)

    def _extract_persons_from_root(self, root: ET.Element) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Internal method to extract persons from XML root element.

        Args:
            root (ET.Element): Root XML element

        Returns:
            Dict[str, Tuple[List[str], List[str]]]: Dictionary with extracted person data
        """
        self.persons_dict = {}

        # Find all person elements
        persons = root.findall('.//tei:person', self.namespace)

        for person in persons:
            person_id = person.get('{http://www.w3.org/XML/1998/namespace}id')

            if person_id:
                surnames = []
                forenames = []

                # Find all persName elements (main and aliases)
                pers_names = person.findall('.//tei:persName', self.namespace)

                for pers_name in pers_names:
                    # Extract surname
                    surname_elem = pers_name.find('tei:surname', self.namespace)
                    if surname_elem is not None and surname_elem.text:
                        surname = surname_elem.text.strip()
                        if surname and surname not in surnames:
                            surnames.append(surname)

                    # Extract forename
                    forename_elem = pers_name.find('tei:forename', self.namespace)
                    if forename_elem is not None and forename_elem.text:
                        forename = forename_elem.text.strip()
                        if forename and forename not in forenames:
                            forenames.append(forename)

                # Store in dictionary
                self.persons_dict[person_id] = (surnames, forenames)

        return self.persons_dict

    def save_to_json(self, output_file_path: str, indent: int = 2) -> None:
        """
        Save the extracted person data to a JSON file.

        Args:
            output_file_path (str): Path where to save the JSON file
            indent (int): JSON indentation level
        """
        # Convert tuple values to lists for JSON serialization
        json_dict = {
            person_id: {
                "surnames": surnames,
                "forenames": forenames
            }
            for person_id, (surnames, forenames) in self.persons_dict.items()
        }

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, indent=indent, ensure_ascii=False)

    def get_json_string(self, indent: int = 2) -> str:
        """
        Get the extracted person data as a JSON string.

        Args:
            indent (int): JSON indentation level

        Returns:
            str: JSON string representation of the data
        """
        # Convert tuple values to lists for JSON serialization
        json_dict = {
            person_id: {
                "surnames": surnames,
                "forenames": forenames
            }
            for person_id, (surnames, forenames) in self.persons_dict.items()
        }

        return json.dumps(json_dict, indent=indent, ensure_ascii=False)

    def print_summary(self) -> None:
        """Print a summary of extracted persons."""
        print(f"Extracted {len(self.persons_dict)} persons:")
        for person_id, (surnames, forenames) in self.persons_dict.items():
            print(f"  {person_id}: {len(surnames)} surnames, {len(forenames)} forenames")


# Example usage
if __name__ == "__main__":
    # Create extractor instance
    extractor = TEIPersonExtractor()
    # Extract persons
    extractor.extract_persons_from_file('persons.xml')

    # Print summary
    extractor.print_summary()

    # Save to JSON file
    # extractor.save_to_json('persons.json')

    # Or get as JSON string
    json_output = extractor.get_json_string()
    print("JSON Output:")
    print(json_output)

    with open('persons.json', 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
