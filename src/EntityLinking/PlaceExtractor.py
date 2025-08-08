import xml.etree.ElementTree as ET
import json
from typing import Dict, Optional

class TEIPlaceExtractor:
    """
    A class to extract place information from TEI XML files and convert to structured JSON.

    The class parses XML files containing place data in TEI format and creates a dictionary
    where each place's xml:id maps to its settlement name.
    """

    def __init__(self):
        self.namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
        self.places_dict = {}

    def extract_places_from_file(self, xml_file_path: str) -> Dict[str, str]:
        """
        Extract place data from an XML file.

        Args:
            xml_file_path (str): Path to the XML file

        Returns:
            Dict[str, str]: Dictionary with place IDs as keys and settlement names as values
        """
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        return self._extract_places_from_root(root)

    def extract_places_from_string(self, xml_string: str) -> Dict[str, str]:
        """
        Extract place data from an XML string.

        Args:
            xml_string (str): XML content as string

        Returns:
            Dict[str, str]: Dictionary with place IDs as keys and settlement names as values
        """
        root = ET.fromstring(xml_string)
        return self._extract_places_from_root(root)

    def _extract_places_from_root(self, root: ET.Element) -> Dict[str, str]:
        """
        Internal method to extract places from XML root element.
        Falls back to district, then country if settlement is not available.

        Args:
            root (ET.Element): Root XML element

        Returns:
            Dict[str, str]: Dictionary with extracted place data
        """
        self.places_dict = {}

        # Find all place elements
        places = root.findall('.//tei:place', self.namespace)

        for place in places:
            place_id = place.get('{http://www.w3.org/XML/1998/namespace}id')

            if place_id:
                place_name = None

                # Try to extract settlement name first
                settlement_elem = place.find('tei:settlement', self.namespace)
                if settlement_elem is not None and settlement_elem.text and settlement_elem.text.strip():
                    place_name = settlement_elem.text.strip()

                # If no settlement, try district
                if not place_name:
                    district_elem = place.find('tei:district', self.namespace)
                    if district_elem is not None and district_elem.text and district_elem.text.strip():
                        place_name = district_elem.text.strip()

                # If no district, try country
                if not place_name:
                    country_elem = place.find('tei:country', self.namespace)
                    if country_elem is not None and country_elem.text and country_elem.text.strip():
                        place_name = country_elem.text.strip()

                # Store the result (could still be None if none of the elements exist)
                self.places_dict[place_id] = place_name

        return self.places_dict

    def extract_places_with_details_from_file(self, xml_file_path: str) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Extract detailed place data from an XML file including district, country, and coordinates.

        Args:
            xml_file_path (str): Path to the XML file

        Returns:
            Dict[str, Dict[str, Optional[str]]]: Dictionary with place IDs as keys and place details as values
        """
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        return self._extract_places_with_details_from_root(root)

    def extract_places_with_details_from_string(self, xml_string: str) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Extract detailed place data from an XML string including district, country, and coordinates.

        Args:
            xml_string (str): XML content as string

        Returns:
            Dict[str, Dict[str, Optional[str]]]: Dictionary with place IDs as keys and place details as values
        """
        root = ET.fromstring(xml_string)
        return self._extract_places_with_details_from_root(root)

    def _extract_places_with_details_from_root(self, root: ET.Element) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Internal method to extract detailed places from XML root element.

        Args:
            root (ET.Element): Root XML element

        Returns:
            Dict[str, Dict[str, Optional[str]]]: Dictionary with extracted detailed place data
        """
        places_details_dict = {}

        # Find all place elements
        places = root.findall('.//tei:place', self.namespace)

        for place in places:
            place_id = place.get('{http://www.w3.org/XML/1998/namespace}id')

            if place_id:
                place_info = {}

                # Extract settlement name
                settlement_elem = place.find('tei:settlement', self.namespace)
                place_info['settlement'] = settlement_elem.text.strip() if settlement_elem is not None and settlement_elem.text else None

                # Extract district
                district_elem = place.find('tei:district', self.namespace)
                place_info['district'] = district_elem.text.strip() if district_elem is not None and district_elem.text else None

                # Extract country
                country_elem = place.find('tei:country', self.namespace)
                place_info['country'] = country_elem.text.strip() if country_elem is not None and country_elem.text else None

                # Extract coordinates
                geo_elem = place.find('.//tei:geo', self.namespace)
                place_info['coordinates'] = geo_elem.text.strip() if geo_elem is not None and geo_elem.text else None

                # Extract geonames URL
                geonames_elem = place.find('.//tei:idno[@subtype="geonames"]', self.namespace)
                place_info['geonames_url'] = geonames_elem.text.strip() if geonames_elem is not None and geonames_elem.text else None

                places_details_dict[place_id] = place_info

        return places_details_dict

    def save_to_json(self, output_file_path: str, indent: int = 2, include_details: bool = False) -> None:
        """
        Save the extracted place data to a JSON file.

        Args:
            output_file_path (str): Path where to save the JSON file
            indent (int): JSON indentation level
            include_details (bool): Whether to include detailed place information
        """
        if include_details:
            # Need to extract detailed data first
            data_to_save = self.extract_places_with_details_from_root(ET.fromstring(self._last_xml_content)) if hasattr(self, '_last_xml_content') else {}
        else:
            data_to_save = self.places_dict

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=indent, ensure_ascii=False)

    def get_json_string(self, indent: int = 2, include_details: bool = False) -> str:
        """
        Get the extracted place data as a JSON string.

        Args:
            indent (int): JSON indentation level
            include_details (bool): Whether to include detailed place information

        Returns:
            str: JSON string representation of the data
        """
        if include_details:
            # Need to extract detailed data first
            data_to_export = self.extract_places_with_details_from_root(ET.fromstring(self._last_xml_content)) if hasattr(self, '_last_xml_content') else {}
        else:
            data_to_export = self.places_dict

        return json.dumps(data_to_export, indent=indent, ensure_ascii=False)

    def print_summary(self) -> None:
        """Print a summary of extracted places."""
        print(f"Extracted {len(self.places_dict)} places:")
        for place_id, settlement in self.places_dict.items():
            print(f"  {place_id}: {settlement if settlement else 'No settlement name'}")

    def get_places_by_settlement(self, settlement_name: str) -> list:
        """
        Get all place IDs that have the specified settlement name.

        Args:
            settlement_name (str): Name of the settlement to search for

        Returns:
            list: List of place IDs with matching settlement name
        """
        return [place_id for place_id, name in self.places_dict.items() if name == settlement_name]


# Example usage
if __name__ == "__main__":
    # Create extractor instance
    extractor = TEIPlaceExtractor()

    # Extract places (simple version - just ID and settlement)
    extractor.extract_places_from_file('localities.xml')

    # Print summary
    extractor.print_summary()

    # Get simple JSON output
    simple_json = extractor.get_json_string()
    print("\nSimple JSON output (ID -> Settlement):")
    print(simple_json)

    # Extract detailed place information
    detailed_places = extractor.extract_places_with_details_from_file('localities.xml')
    print("\nDetailed JSON output:")
    print(json.dumps(detailed_places, indent=2, ensure_ascii=False))

    # Save to files
    extractor.save_to_json('places_simple.json')
    extractor.save_to_json('places_detailed.json', include_details=True)