import xml.etree.ElementTree as ET
from collections import defaultdict
import Levenshtein
from typing import Dict, List, Set, Tuple

class LocalityAmbiguityAnalyzer:
    def __init__(self, xml_file_path: str):
        self.xml_file_path = xml_file_path
        self.places = {}  # id -> {'settlement': str, 'district': str, 'country': str}
        self.all_names = set()  # All unique location names
        self.settlement_to_ids = defaultdict(set)  # settlement -> set of place ids
        self.district_to_ids = defaultdict(set)  # district -> set of place ids
        self.country_to_ids = defaultdict(set)  # country -> set of place ids
        self.name_to_ids = defaultdict(set)  # any location name -> set of place ids
        self.name_to_types = defaultdict(set)  # name -> set of types (settlement, district, country)

    def parse_xml(self):
        """Parse the XML file and extract all places with their location components."""
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()

        # Define namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # Find all place elements
        places = root.findall('.//tei:place', ns)

        for place in places:
            place_id = place.get('{http://www.w3.org/XML/1998/namespace}id')
            if not place_id:
                continue

            # Initialize place data
            place_data = {'settlement': None, 'district': None, 'country': None}

            # Extract settlement
            settlement_elem = place.find('tei:settlement', ns)
            if settlement_elem is not None and settlement_elem.text:
                settlement = settlement_elem.text.strip()
                place_data['settlement'] = settlement
                self.all_names.add(settlement)
                self.settlement_to_ids[settlement.lower()].add(place_id)
                self.name_to_ids[settlement.lower()].add(place_id)
                self.name_to_types[settlement.lower()].add('settlement')

            # Extract district
            district_elem = place.find('tei:district', ns)
            if district_elem is not None and district_elem.text:
                district = district_elem.text.strip()
                place_data['district'] = district
                self.all_names.add(district)
                self.district_to_ids[district.lower()].add(place_id)
                self.name_to_ids[district.lower()].add(place_id)
                self.name_to_types[district.lower()].add('district')

            # Extract country
            country_elem = place.find('tei:country', ns)
            if country_elem is not None and country_elem.text:
                country = country_elem.text.strip()
                place_data['country'] = country
                self.all_names.add(country)
                self.country_to_ids[country.lower()].add(place_id)
                self.name_to_ids[country.lower()].add(place_id)
                self.name_to_types[country.lower()].add('country')

            self.places[place_id] = place_data

    def analyze_exact_matches(self):
        """Analyze exact name matches that could cause ambiguity."""
        print("=== EXACT MATCH AMBIGUITY ANALYSIS ===\n")

        # Settlement ambiguities
        settlement_ambiguities = {name: ids for name, ids in self.settlement_to_ids.items() if len(ids) > 1}
        print(f"Settlements shared by multiple places: {len(settlement_ambiguities)}")

        total_settlement_conflicts = sum(len(ids) for ids in settlement_ambiguities.values())
        print(f"Total place instances involved in settlement conflicts: {total_settlement_conflicts}")

        # Show top ambiguous settlements
        sorted_settlement_amb = sorted(settlement_ambiguities.items(), key=lambda x: len(x[1]), reverse=True)
        print("\nTop 10 most ambiguous settlements:")
        for settlement, ids in sorted_settlement_amb[:10]:
            print(f"  '{settlement}': {len(ids)} places ({', '.join(sorted(ids))})")
            # Show which districts/countries these settlements are in
            districts = set()
            countries = set()
            for place_id in ids:
                if self.places[place_id]['district']:
                    districts.add(self.places[place_id]['district'])
                if self.places[place_id]['country']:
                    countries.add(self.places[place_id]['country'])
            print(f"    Districts: {', '.join(sorted(districts)) if districts else 'None'}")
            print(f"    Countries: {', '.join(sorted(countries)) if countries else 'None'}")

        # District ambiguities
        district_ambiguities = {name: ids for name, ids in self.district_to_ids.items() if len(ids) > 1}
        print(f"\nDistricts shared by multiple places: {len(district_ambiguities)}")

        total_district_conflicts = sum(len(ids) for ids in district_ambiguities.values())
        print(f"Total place instances involved in district conflicts: {total_district_conflicts}")

        # Show top ambiguous districts
        sorted_district_amb = sorted(district_ambiguities.items(), key=lambda x: len(x[1]), reverse=True)
        print("\nTop 10 most ambiguous districts:")
        for district, ids in sorted_district_amb[:10]:
            print(f"  '{district}': {len(ids)} places ({', '.join(sorted(ids))})")

        # Country ambiguities
        country_ambiguities = {name: ids for name, ids in self.country_to_ids.items() if len(ids) > 1}
        print(f"\nCountries shared by multiple places: {len(country_ambiguities)}")

        total_country_conflicts = sum(len(ids) for ids in country_ambiguities.values())
        print(f"Total place instances involved in country conflicts: {total_country_conflicts}")

        # Show countries (typically many places per country)
        sorted_country_amb = sorted(country_ambiguities.items(), key=lambda x: len(x[1]), reverse=True)
        print("\nTop 5 countries by number of places:")
        for country, ids in sorted_country_amb[:5]:
            print(f"  '{country}': {len(ids)} places")

    def analyze_cross_category_conflicts(self):
        """Analyze cases where the same name is used across different location types."""
        print("\n=== CROSS-CATEGORY CONFLICTS ===\n")

        cross_conflicts = []

        for name in self.all_names:
            name_lower = name.lower()
            types_for_name = self.name_to_types[name_lower]

            if len(types_for_name) > 1:
                # This name is used as multiple types
                settlement_ids = self.settlement_to_ids.get(name_lower, set())
                district_ids = self.district_to_ids.get(name_lower, set())
                country_ids = self.country_to_ids.get(name_lower, set())

                conflict_info = {
                    'name': name,
                    'types': types_for_name,
                    'settlement_count': len(settlement_ids),
                    'district_count': len(district_ids),
                    'country_count': len(country_ids),
                    'total_places': len(settlement_ids | district_ids | country_ids)
                }
                cross_conflicts.append(conflict_info)

        print(f"Names used across multiple location types: {len(cross_conflicts)}")

        # Sort by total number of places affected
        cross_conflicts.sort(key=lambda x: x['total_places'], reverse=True)

        print("\nTop 15 cross-category conflicts:")
        for conflict in cross_conflicts[:15]:
            name = conflict['name']
            types = ', '.join(sorted(conflict['types']))
            print(f"  '{name}' used as: {types}")
            if conflict['settlement_count'] > 0:
                print(f"    As settlement: {conflict['settlement_count']} places")
            if conflict['district_count'] > 0:
                print(f"    As district: {conflict['district_count']} places")
            if conflict['country_count'] > 0:
                print(f"    As country: {conflict['country_count']} places")

    def analyze_levenshtein_ambiguity(self, threshold: int = 1):
        """Analyze ambiguity using Levenshtein distance."""
        print(f"\n=== LEVENSHTEIN DISTANCE AMBIGUITY (threshold: {threshold}) ===\n")

        all_names_list = list(self.all_names)
        similar_pairs = []

        for i, name1 in enumerate(all_names_list):
            for j, name2 in enumerate(all_names_list[i+1:], i+1):
                distance = Levenshtein.distance(name1.lower(), name2.lower())
                if 0 < distance <= threshold:
                    # Get all place IDs associated with these names
                    ids1 = self.name_to_ids[name1.lower()]
                    ids2 = self.name_to_ids[name2.lower()]

                    # Get the types for these names
                    types1 = self.name_to_types[name1.lower()]
                    types2 = self.name_to_types[name2.lower()]

                    # Only count as ambiguous if they refer to different places
                    if ids1 != ids2:
                        similar_pairs.append((name1, name2, distance, ids1, ids2, types1, types2))

        print(f"Similar location name pairs (Levenshtein distance ≤ {threshold}): {len(similar_pairs)}")

        # Sort by distance, then by name
        similar_pairs.sort(key=lambda x: (x[2], x[0], x[1]))

        print(f"\nTop 20 similar location name pairs:")
        for name1, name2, dist, ids1, ids2, types1, types2 in similar_pairs[:20]:
            print(f"  '{name1}' ↔ '{name2}' (distance: {dist})")
            print(f"    '{name1}' → {len(ids1)} places ({', '.join(sorted(types1))}): {', '.join(sorted(ids1))}")
            print(f"    '{name2}' → {len(ids2)} places ({', '.join(sorted(types2))}): {', '.join(sorted(ids2))}")

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
                        types1 = self.name_to_types[name1.lower()]
                        types2 = self.name_to_types[name2.lower()]
                        if ids1 != ids2:  # Different places
                            substring_matches.append((name1, name2, ids1, ids2, types1, types2))

        print(f"Substring relationships between location names: {len(substring_matches)}")

        # Remove duplicates and sort
        unique_substring_matches = []
        seen = set()
        for name1, name2, ids1, ids2, types1, types2 in substring_matches:
            pair = tuple(sorted([name1.lower(), name2.lower()]))
            if pair not in seen:
                seen.add(pair)
                unique_substring_matches.append((name1, name2, ids1, ids2, types1, types2))

        print(f"\nTop 15 substring relationships:")
        for name1, name2, ids1, ids2, types1, types2 in unique_substring_matches[:15]:
            shorter, longer = (name1, name2) if len(name1) < len(name2) else (name2, name1)
            print(f"  '{shorter}' ⊆ '{longer}'")
            print(f"    '{name1}' → {len(ids1)} places ({', '.join(sorted(types1))})")
            print(f"    '{name2}' → {len(ids2)} places ({', '.join(sorted(types2))})")

    def analyze_hierarchical_conflicts(self):
        """Analyze conflicts within the settlement-district-country hierarchy."""
        print("\n=== HIERARCHICAL CONFLICTS ===\n")

        # Find places where settlement name matches district or country name of other places
        hierarchical_conflicts = []

        for place_id, place_data in self.places.items():
            settlement = place_data['settlement']
            district = place_data['district']
            country = place_data['country']

            conflicts = []

            # Check if this place's settlement matches other places' districts
            if settlement:
                other_district_ids = self.district_to_ids.get(settlement.lower(), set())
                if other_district_ids and place_id not in other_district_ids:
                    conflicts.append(f"Settlement '{settlement}' matches district in places: {', '.join(sorted(other_district_ids))}")

                # Check if this place's settlement matches other places' countries
                other_country_ids = self.country_to_ids.get(settlement.lower(), set())
                if other_country_ids and place_id not in other_country_ids:
                    conflicts.append(f"Settlement '{settlement}' matches country in places: {', '.join(sorted(other_country_ids))}")

            # Check if this place's district matches other places' countries
            if district:
                other_country_ids = self.country_to_ids.get(district.lower(), set())
                if other_country_ids and place_id not in other_country_ids:
                    conflicts.append(f"District '{district}' matches country in places: {', '.join(sorted(other_country_ids))}")

            if conflicts:
                hierarchical_conflicts.append((place_id, place_data, conflicts))

        print(f"Places with hierarchical naming conflicts: {len(hierarchical_conflicts)}")

        print("\nTop 10 hierarchical conflicts:")
        for place_id, place_data, conflicts in hierarchical_conflicts[:10]:
            settlement = place_data['settlement'] or 'None'
            district = place_data['district'] or 'None'
            country = place_data['country'] or 'None'
            print(f"  {place_id}: {settlement} / {district} / {country}")
            for conflict in conflicts:
                print(f"    {conflict}")

    def print_summary_statistics(self):
        """Print overall statistics about the dataset."""
        print("\n=== SUMMARY STATISTICS ===\n")

        total_places = len(self.places)
        total_unique_names = len(self.all_names)

        # Count places by component availability
        places_with_settlement = sum(1 for p in self.places.values() if p['settlement'])
        places_with_district = sum(1 for p in self.places.values() if p['district'])
        places_with_country = sum(1 for p in self.places.values() if p['country'])

        places_with_all_three = sum(1 for p in self.places.values()
                                    if p['settlement'] and p['district'] and p['country'])

        print(f"Total places: {total_places}")
        print(f"Places with settlement: {places_with_settlement} ({places_with_settlement/total_places*100:.1f}%)")
        print(f"Places with district: {places_with_district} ({places_with_district/total_places*100:.1f}%)")
        print(f"Places with country: {places_with_country} ({places_with_country/total_places*100:.1f}%)")
        print(f"Places with all three components: {places_with_all_three} ({places_with_all_three/total_places*100:.1f}%)")

        print(f"Total unique location names: {total_unique_names}")
        print(f"Unique settlements: {len(self.settlement_to_ids)}")
        print(f"Unique districts: {len(self.district_to_ids)}")
        print(f"Unique countries: {len(self.country_to_ids)}")

        # Ambiguity potential
        ambiguous_names = sum(1 for ids in self.name_to_ids.values() if len(ids) > 1)
        print(f"Names that could cause ambiguity: {ambiguous_names} ({ambiguous_names/total_unique_names*100:.1f}%)")

    def run_analysis(self):
        """Run the complete ambiguity analysis."""
        print("Parsing XML file...")
        self.parse_xml()

        self.print_summary_statistics()
        self.analyze_exact_matches()
        self.analyze_cross_category_conflicts()
        self.analyze_hierarchical_conflicts()
        self.analyze_levenshtein_ambiguity(threshold=1)
        self.analyze_levenshtein_ambiguity(threshold=2)
        self.analyze_partial_matches()

# Usage example:
if __name__ == "__main__":
    # Initialize analyzer with your XML file
    analyzer = LocalityAmbiguityAnalyzer("localities.xml")  # Replace with your actual file path
    analyzer.run_analysis()