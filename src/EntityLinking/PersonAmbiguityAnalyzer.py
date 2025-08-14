import xml.etree.ElementTree as ET
from collections import defaultdict
import Levenshtein
from typing import Dict, List, Set, Tuple

class PersonAmbiguityAnalyzer:
    def __init__(self, xml_file_path: str):
        self.xml_file_path = xml_file_path
        self.persons = {}  # id -> {'main': (surname, forename), 'aliases': [(surname, forename), ...]}
        self.all_names = set()  # All unique names (surnames and forenames)
        self.surname_to_ids = defaultdict(set)  # surname -> set of person ids
        self.forename_to_ids = defaultdict(set)  # forename -> set of person ids
        self.name_to_ids = defaultdict(set)  # any name -> set of person ids

    def parse_xml(self):
        """Parse the XML file and extract all persons with their names and aliases."""
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()

        # Define namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # Find all person elements
        persons = root.findall('.//tei:person', ns)

        for person in persons:
            person_id = person.get('{http://www.w3.org/XML/1998/namespace}id')
            if not person_id:
                continue

            # Initialize person data
            self.persons[person_id] = {'main': None, 'aliases': []}

            # Find all persName elements
            persNames = person.findall('tei:persName', ns)

            for persName in persNames:
                surname_elem = persName.find('tei:surname', ns)
                forename_elem = persName.find('tei:forename', ns)

                surname = surname_elem.text.strip() if surname_elem is not None and surname_elem.text else ""
                forename = forename_elem.text.strip() if forename_elem is not None and forename_elem.text else ""

                # Skip if both are empty
                if not surname and not forename:
                    continue

                name_tuple = (surname, forename)

                # Check if this is the main name or an alias
                if persName.get('type') == 'main':
                    self.persons[person_id]['main'] = name_tuple
                else:
                    self.persons[person_id]['aliases'].append(name_tuple)

                # Add to our indices
                if surname:
                    self.all_names.add(surname)
                    self.surname_to_ids[surname.lower()].add(person_id)
                    self.name_to_ids[surname.lower()].add(person_id)

                if forename:
                    self.all_names.add(forename)
                    self.forename_to_ids[forename.lower()].add(person_id)
                    self.name_to_ids[forename.lower()].add(person_id)

    def analyze_exact_matches(self):
        """Analyze exact name matches that could cause ambiguity."""
        print("=== EXACT MATCH AMBIGUITY ANALYSIS ===\n")

        # Surname ambiguities
        surname_ambiguities = {name: ids for name, ids in self.surname_to_ids.items() if len(ids) > 1}
        print(f"Surnames shared by multiple persons: {len(surname_ambiguities)}")

        total_surname_conflicts = sum(len(ids) for ids in surname_ambiguities.values())
        print(f"Total person instances involved in surname conflicts: {total_surname_conflicts}")

        # Show top ambiguous surnames
        sorted_surname_amb = sorted(surname_ambiguities.items(), key=lambda x: len(x[1]), reverse=True)
        print("\nTop 10 most ambiguous surnames:")
        for surname, ids in sorted_surname_amb[:10]:
            print(f"  '{surname}': {len(ids)} persons ({', '.join(sorted(ids))})")

        # Forename ambiguities
        forename_ambiguities = {name: ids for name, ids in self.forename_to_ids.items() if len(ids) > 1}
        print(f"\nForenames shared by multiple persons: {len(forename_ambiguities)}")

        total_forename_conflicts = sum(len(ids) for ids in forename_ambiguities.values())
        print(f"Total person instances involved in forename conflicts: {total_forename_conflicts}")

        # Show top ambiguous forenames
        sorted_forename_amb = sorted(forename_ambiguities.items(), key=lambda x: len(x[1]), reverse=True)
        print("\nTop 10 most ambiguous forenames:")
        for forename, ids in sorted_forename_amb[:10]:
            print(f"  '{forename}': {len(ids)} persons ({', '.join(sorted(ids))})")

        # Cross-category conflicts (surname matches forename)
        cross_conflicts = []
        for surname in self.surname_to_ids:
            if surname in self.forename_to_ids:
                surname_ids = self.surname_to_ids[surname]
                forename_ids = self.forename_to_ids[surname]
                if surname_ids != forename_ids:  # Different sets of people
                    cross_conflicts.append((surname, surname_ids, forename_ids))

        print(f"\nCross-category conflicts (name used as both surname and forename): {len(cross_conflicts)}")
        for name, surname_ids, forename_ids in cross_conflicts[:10]:
            print(f"  '{name}': surname for {len(surname_ids)} persons, forename for {len(forename_ids)} persons")

    def analyze_levenshtein_ambiguity(self, threshold: int = 1):
        """Analyze ambiguity using Levenshtein distance."""
        print(f"\n=== LEVENSHTEIN DISTANCE AMBIGUITY (threshold: {threshold}) ===\n")

        all_names_list = list(self.all_names)
        similar_pairs = []

        for i, name1 in enumerate(all_names_list):
            for j, name2 in enumerate(all_names_list[i+1:], i+1):
                distance = Levenshtein.distance(name1.lower(), name2.lower())
                if 0 < distance <= threshold:
                    # Get all person IDs associated with these names
                    ids1 = self.name_to_ids[name1.lower()]
                    ids2 = self.name_to_ids[name2.lower()]

                    # Only count as ambiguous if they refer to different persons
                    if ids1 != ids2 and not ids1.isdisjoint(ids2) == False:
                        similar_pairs.append((name1, name2, distance, ids1, ids2))

        print(f"Similar name pairs (Levenshtein distance ≤ {threshold}): {len(similar_pairs)}")

        # Sort by distance, then by name
        similar_pairs.sort(key=lambda x: (x[2], x[0], x[1]))

        print(f"\nTop 20 similar name pairs:")
        for name1, name2, dist, ids1, ids2 in similar_pairs[:20]:
            print(f"  '{name1}' ↔ '{name2}' (distance: {dist})")
            print(f"    '{name1}' → {len(ids1)} persons: {', '.join(sorted(ids1))}")
            print(f"    '{name2}' → {len(ids2)} persons: {', '.join(sorted(ids2))}")

    def analyze_partial_matches(self):
        """Analyze cases where partial matching could cause issues."""
        print("\n=== PARTIAL MATCH ANALYSIS ===\n")

        # Find names that are substrings of other names
        substring_matches = []
        all_names_list = list(self.all_names)

        for i, name1 in enumerate(all_names_list):
            for j, name2 in enumerate(all_names_list):
                if i != j and len(name1) >= 3 and len(name2) >= 3:
                    if name1.lower() in name2.lower() or name2.lower() in name1.lower():
                        ids1 = self.name_to_ids[name1.lower()]
                        ids2 = self.name_to_ids[name2.lower()]
                        if ids1 != ids2:  # Different persons
                            substring_matches.append((name1, name2, ids1, ids2))

        print(f"Substring relationships between names: {len(substring_matches)}")

        # Remove duplicates and sort
        unique_substring_matches = []
        seen = set()
        for name1, name2, ids1, ids2 in substring_matches:
            pair = tuple(sorted([name1.lower(), name2.lower()]))
            if pair not in seen:
                seen.add(pair)
                unique_substring_matches.append((name1, name2, ids1, ids2))

        print(f"\nTop 15 substring relationships:")
        for name1, name2, ids1, ids2 in unique_substring_matches[:15]:
            shorter, longer = (name1, name2) if len(name1) < len(name2) else (name2, name1)
            print(f"  '{shorter}' ⊆ '{longer}'")
            print(f"    '{name1}' → {len(ids1)} persons, '{name2}' → {len(ids2)} persons")

    def print_summary_statistics(self):
        """Print overall statistics about the dataset."""
        print("\n=== SUMMARY STATISTICS ===\n")

        total_persons = len(self.persons)
        total_unique_names = len(self.all_names)

        # Count total name instances (including aliases)
        total_name_instances = 0
        persons_with_aliases = 0

        for person_id, data in self.persons.items():
            if data['main']:
                total_name_instances += 1
            total_name_instances += len(data['aliases'])
            if data['aliases']:
                persons_with_aliases += 1

        print(f"Total persons: {total_persons}")
        print(f"Persons with aliases: {persons_with_aliases} ({persons_with_aliases/total_persons*100:.1f}%)")
        print(f"Total unique names: {total_unique_names}")
        print(f"Total name instances: {total_name_instances}")
        print(f"Average name instances per person: {total_name_instances/total_persons:.1f}")

        # Ambiguity potential
        ambiguous_names = sum(1 for ids in self.name_to_ids.values() if len(ids) > 1)
        print(f"Names that could cause ambiguity: {ambiguous_names} ({ambiguous_names/total_unique_names*100:.1f}%)")

    def run_analysis(self):
        """Run the complete ambiguity analysis."""
        print("Parsing XML file...")
        self.parse_xml()

        self.print_summary_statistics()
        self.analyze_exact_matches()
        self.analyze_levenshtein_ambiguity(threshold=1)
        self.analyze_levenshtein_ambiguity(threshold=2)
        self.analyze_partial_matches()

# Usage example:
if __name__ == "__main__":
    # Initialize analyzer with your XML file
    analyzer = PersonAmbiguityAnalyzer("persons.xml")  # Replace with your actual file path
    analyzer.run_analysis()