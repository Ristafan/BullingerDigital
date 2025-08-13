import json
from collections import defaultdict, Counter


class ResultComparer:
    def __init__(self, predictions_and_labels_json):
        self.predictions_and_labels_json = predictions_and_labels_json
        self.tokenized_sentences = {}
        self.labels = {}
        self.predictions = {}

        # Results storage
        self.sentence_results = {}
        self.file_results = defaultdict(lambda: {
            'exact_mismatches': {'total': 0, 'person': 0, 'location': 0},
            'flexible_mismatches': {'total': 0, 'person': 0, 'location': 0},
            'additional_entities': defaultdict(int),
            'missed_entities': defaultdict(int),
            'correct_entities': defaultdict(int)
        })
        self.testset_results = {
            'exact_mismatches': {'total': 0, 'person': 0, 'location': 0},
            'flexible_mismatches': {'total': 0, 'person': 0, 'location': 0},
            'additional_entities': defaultdict(int),
            'missed_entities': defaultdict(int),
            'correct_entities': defaultdict(int)
        }

        self.read_labels_and_predictions()

    def evaluate(self):
        """Main evaluation function that processes all sentences and aggregates results."""
        print("Starting evaluation...")

        for filename_and_sentence_idx in self.tokenized_sentences.keys():
            label_entities = self.labels[filename_and_sentence_idx]
            predicted_entities = self.predictions[filename_and_sentence_idx]
            tokenized_sentence = self.tokenized_sentences[filename_and_sentence_idx]

            # Get filename for file-level aggregation
            filename = filename_and_sentence_idx.split('_')[0] + '.xml'

            # 1. Count mismatches (exact and flexible)
            exact_mismatches = self.count_exact_mismatches(predicted_entities, label_entities)
            flexible_mismatches = self.count_flexible_mismatches(predicted_entities, label_entities)

            # 2. Generate colored sentence
            colored_sentence = self.generate_colored_sentence(
                tokenized_sentence, label_entities, predicted_entities
            )

            # 3. Analyze entities
            entity_analysis = self.analyze_entities(predicted_entities, label_entities, tokenized_sentence)

            # Store sentence-level results
            self.sentence_results[filename_and_sentence_idx] = {
                'exact_mismatches': exact_mismatches,
                'flexible_mismatches': flexible_mismatches,
                'colored_sentence': colored_sentence,
                'entity_analysis': entity_analysis
            }

            # Aggregate file-level results
            self._aggregate_to_file_level(filename, exact_mismatches, flexible_mismatches, entity_analysis)

            # Aggregate test set level results
            self._aggregate_to_testset_level(exact_mismatches, flexible_mismatches, entity_analysis)

        self._print_results()
        return self.sentence_results, dict(self.file_results), self.testset_results

    def convert_char_span_to_token_span(self, char_start, char_end, original_sentence, tokenized_sentence):
        """Convert character-based span to token-based span."""
        substring_of_sentence = original_sentence[char_start:char_end]

        # If substring is made up of multiple words, we need to handle it carefully
        if len(substring_of_sentence.split(' ')) > 1:
            tokens = substring_of_sentence.split(' ')

            if tokens[0] in tokenized_sentence:
                start_token = tokenized_sentence.index(tokens[0])
                end_token = start_token + len(tokens) - 1

        if substring_of_sentence in tokenized_sentence:
            start_token = tokenized_sentence.index(substring_of_sentence)
            end_token = start_token + substring_of_sentence.count(' ')

        return [start_token, end_token]

    def fix_char_based_predictions(self, entities, original_sentence, tokenized_sentence, max_token_span=5):
        """Fix predictions that use character spans instead of token spans."""
        fixed_entities = []

        for entity in entities:
            start, end, entity_type = entity

            # Check if this looks like a character span (large span or suspicious range)
            span_length = end - start + 1

            if span_length > max_token_span or end >= len(tokenized_sentence):
                # This looks like a character-based span, try to convert
                converted_span = self.convert_char_span_to_token_span(
                    start, end, original_sentence, tokenized_sentence
                )

                if converted_span is not None:
                    start_token, end_token = converted_span
                    fixed_entities.append([start_token, end_token, entity_type])
                    # print(f"Fixed char span [{start}, {end}] -> token span [{start_token}, {end_token}] for '{entity_type}'")
                else:
                    # Could not convert, keep original but mark as suspicious
                    print(f"Warning: Could not convert suspicious span [{start}, {end}] for '{entity_type}' - keeping original")
                    fixed_entities.append(entity)
            else:
                # Normal token-based span
                fixed_entities.append(entity)

        return fixed_entities

    def read_labels_and_predictions(self):
        """Read and parse the labels JSON file."""
        with open(self.predictions_and_labels_json, 'r', encoding="utf-8") as file:
            sentences = json.load(file)

        for sentence in sentences:
            filename_and_sentence_idx = sentence["sentence_pair"]
            original_sentence = sentence["original_sentence"]
            tokenized_sentence = original_sentence.split(" ")
            labelled_entities = sentence["labelled_entities"]
            entities = sentence["entities"]

            # Correct labels and entities format
            for e in labelled_entities:
                if e[2].startswith("person"):
                    e[2] = "person"
                elif e[2].startswith("location"):
                    e[2] = "location"

            entities = [[e['start'], e['end'], e['label']] for e in entities]


            # Fix character-based predictions
            fixed_entities = self.fix_char_based_predictions(
                entities, original_sentence, tokenized_sentence
            )

            # Store tokenized sentences
            self.tokenized_sentences[filename_and_sentence_idx] = tokenized_sentence
            # Store labels
            self.labels[filename_and_sentence_idx] = labelled_entities
            # Store predictions (now fixed)
            self.predictions[filename_and_sentence_idx] = fixed_entities

    def count_exact_mismatches(self, predicted_entities, label_entities):
        """Count mismatches with exact token position matching."""
        total_mismatch = len(label_entities) - len(predicted_entities)

        predicted_persons = [e for e in predicted_entities if e[2].lower() == "person"]
        label_persons = [e for e in label_entities if e[2].lower() == "person"]
        person_mismatch = len(label_persons) - len(predicted_persons)

        predicted_locations = [e for e in predicted_entities if e[2].lower() == "location"]
        label_locations = [e for e in label_entities if e[2].lower() == "location"]
        location_mismatch = len(label_locations) - len(predicted_locations)

        return {
            'total': total_mismatch,
            'person': person_mismatch,
            'location': location_mismatch
        }

    def count_flexible_mismatches(self, predicted_entities, label_entities):
        """Count mismatches considering flexible entity boundaries."""
        # Create sets of entity spans for comparison
        label_spans = set()
        pred_spans = set()

        for entity in label_entities:
            start, end, entity_type = entity
            label_spans.add((start, end, entity_type.lower()))

        for entity in predicted_entities:
            start, end, entity_type = entity
            pred_spans.add((start, end, entity_type.lower()))

        # Find matches considering flexible boundaries
        matched_labels = set()
        matched_preds = set()

        for label_span in label_spans:
            for pred_span in pred_spans:
                if self._entities_match_flexible(label_span, pred_span):
                    matched_labels.add(label_span)
                    matched_preds.add(pred_span)

        # Calculate mismatches by entity type
        unmatched_labels = label_spans - matched_labels
        unmatched_preds = pred_spans - matched_preds

        person_missed = len([e for e in unmatched_labels if e[2] == "person"])
        person_extra = len([e for e in unmatched_preds if e[2] == "person"])
        location_missed = len([e for e in unmatched_labels if e[2] == "location"])
        location_extra = len([e for e in unmatched_preds if e[2] == "location"])

        return {
            'total': len(unmatched_labels) - len(unmatched_preds),
            'person': person_missed - person_extra,
            'location': location_missed - location_extra
        }

    def _entities_match_flexible(self, label_span, pred_span):
        """Check if entities match considering flexible boundaries."""
        label_start, label_end, label_type = label_span
        pred_start, pred_end, pred_type = pred_span

        if label_type != pred_type:
            return False

        # Exact match
        if label_start == pred_start and label_end == pred_end:
            return True

        # Check if prediction spans cover the same entity range
        # (handling cases like "Gessner , Konrad" vs separate "Gessner" and "Konrad")
        if (pred_start <= label_start <= pred_end or
                pred_start <= label_end <= pred_end or
                label_start <= pred_start <= label_end):
            return True

        return False

    def generate_colored_sentence(self, tokenized_sentence, label_entities, predicted_entities):
        """Generate HTML-colored sentence showing entity matches."""
        # Create token-to-color mapping
        token_colors = [''] * len(tokenized_sentence)

        # Mark labeled entities (blue)
        for start, end, entity_type in label_entities:
            color_class = 'label-person' if entity_type.lower() == 'person' else 'label-location'
            for i in range(start, min(end + 1, len(tokenized_sentence))):
                if token_colors[i] == '':
                    token_colors[i] = color_class

        # Mark predicted entities and check for matches
        matched_predictions = set()
        for pred_start, pred_end, pred_type in predicted_entities:
            # Check if this prediction matches any label
            is_match = False
            for label_start, label_end, label_type in label_entities:
                if self._entities_match_flexible((label_start, label_end, label_type.lower()),
                                                 (pred_start, pred_end, pred_type.lower())):
                    is_match = True
                    matched_predictions.add((pred_start, pred_end, pred_type))
                    break

            if is_match:
                # Correct prediction (green)
                for i in range(pred_start, min(pred_end + 1, len(tokenized_sentence))):
                    token_colors[i] = 'correct'
            else:
                # Extra prediction (orange)
                color_class = 'extra-person' if pred_type.lower() == 'person' else 'extra-location'
                for i in range(pred_start, min(pred_end + 1, len(tokenized_sentence))):
                    if token_colors[i].startswith('label-'):
                        continue  # Keep label color if it was missed
                    token_colors[i] = color_class

        # Build HTML string
        html_parts = []
        current_color = ''

        for i, token in enumerate(tokenized_sentence):
            if token_colors[i] != current_color:
                if current_color:
                    html_parts.append('</span>')
                current_color = token_colors[i]
                if current_color:
                    html_parts.append(f'<span class="{current_color}">')

            html_parts.append(token)
            if i < len(tokenized_sentence) - 1:
                html_parts.append(' ')

        if current_color:
            html_parts.append('</span>')

        return ''.join(html_parts)

    def analyze_entities(self, predicted_entities, label_entities, tokenized_sentence):
        """Analyze entities to categorize them as additional, missed, or correct."""
        additional = []
        missed = []
        correct = []

        # Convert entities to comparable format
        label_set = set()
        pred_set = set()

        for entity in label_entities:
            start, end, entity_type = entity
            entity_text = ' '.join(tokenized_sentence[start:end+1])
            label_set.add((entity_text, entity_type.lower()))

        for entity in predicted_entities:
            start, end, entity_type = entity
            if start < len(tokenized_sentence) and end < len(tokenized_sentence):
                entity_text = ' '.join(tokenized_sentence[start:end+1])
                pred_set.add((entity_text, entity_type.lower()))

        # Find matches using flexible matching
        matched_labels = set()
        matched_preds = set()

        for label_entity in label_entities:
            label_start, label_end, label_type = label_entity
            label_text = ' '.join(tokenized_sentence[label_start:label_end+1])

            for pred_entity in predicted_entities:
                pred_start, pred_end, pred_type = pred_entity
                if (pred_start < len(tokenized_sentence) and pred_end < len(tokenized_sentence) and
                        self._entities_match_flexible((label_start, label_end, label_type.lower()),
                                                      (pred_start, pred_end, pred_type.lower()))):
                    pred_text = ' '.join(tokenized_sentence[pred_start:pred_end+1])
                    correct.append((label_text, label_type.lower()))
                    matched_labels.add((label_text, label_type.lower()))
                    matched_preds.add((pred_text, pred_type.lower()))
                    break

        # Find additional entities (predicted but not labeled)
        for entity in predicted_entities:
            start, end, entity_type = entity
            if start < len(tokenized_sentence) and end < len(tokenized_sentence):
                entity_text = ' '.join(tokenized_sentence[start:end+1])
                if (entity_text, entity_type.lower()) not in matched_preds:
                    additional.append((entity_text, entity_type.lower()))

        # Find missed entities (labeled but not predicted)
        for entity in label_entities:
            start, end, entity_type = entity
            entity_text = ' '.join(tokenized_sentence[start:end+1])
            if (entity_text, entity_type.lower()) not in matched_labels:
                missed.append((entity_text, entity_type.lower()))

        return {
            'additional': additional,
            'missed': missed,
            'correct': correct
        }

    def _aggregate_to_file_level(self, filename, exact_mismatches, flexible_mismatches, entity_analysis):
        """Aggregate results to file level."""
        file_result = self.file_results[filename]

        # Aggregate mismatches
        for key in exact_mismatches:
            file_result['exact_mismatches'][key] += exact_mismatches[key]
            file_result['flexible_mismatches'][key] += flexible_mismatches[key]

        # Aggregate entity counts
        for entity_text, entity_type in entity_analysis['additional']:
            file_result['additional_entities'][f"{entity_text} ({entity_type})"] += 1

        for entity_text, entity_type in entity_analysis['missed']:
            file_result['missed_entities'][f"{entity_text} ({entity_type})"] += 1

        for entity_text, entity_type in entity_analysis['correct']:
            file_result['correct_entities'][f"{entity_text} ({entity_type})"] += 1

    def _aggregate_to_testset_level(self, exact_mismatches, flexible_mismatches, entity_analysis):
        """Aggregate results to test set level."""
        # Aggregate mismatches
        for key in exact_mismatches:
            self.testset_results['exact_mismatches'][key] += exact_mismatches[key]
            self.testset_results['flexible_mismatches'][key] += flexible_mismatches[key]

        # Aggregate entity counts
        for entity_text, entity_type in entity_analysis['additional']:
            self.testset_results['additional_entities'][f"{entity_text} ({entity_type})"] += 1

        for entity_text, entity_type in entity_analysis['missed']:
            self.testset_results['missed_entities'][f"{entity_text} ({entity_type})"] += 1

        for entity_text, entity_type in entity_analysis['correct']:
            self.testset_results['correct_entities'][f"{entity_text} ({entity_type})"] += 1

    def calculate_metrics(self, correct_count, additional_count, missed_count):
        """Calculate precision, recall, and F1 score."""
        if correct_count + additional_count == 0:
            precision = 0.0
        else:
            precision = correct_count / (correct_count + additional_count)

        if correct_count + missed_count == 0:
            recall = 0.0
        else:
            recall = correct_count / (correct_count + missed_count)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _print_results(self):
        """Print comprehensive evaluation results."""
        print("\n" + "="*80)
        print("NAMED ENTITY RECOGNITION EVALUATION RESULTS")
        print("="*80)

        # Calculate test set metrics
        testset_correct = sum(self.testset_results['correct_entities'].values())
        testset_additional = sum(self.testset_results['additional_entities'].values())
        testset_missed = sum(self.testset_results['missed_entities'].values())
        testset_metrics = self.calculate_metrics(testset_correct, testset_additional, testset_missed)

        # Calculate person-specific metrics for test set
        testset_person_correct = sum(count for entity, count in self.testset_results['correct_entities'].items() if '(person)' in entity)
        testset_person_additional = sum(count for entity, count in self.testset_results['additional_entities'].items() if '(person)' in entity)
        testset_person_missed = sum(count for entity, count in self.testset_results['missed_entities'].items() if '(person)' in entity)
        testset_person_metrics = self.calculate_metrics(testset_person_correct, testset_person_additional, testset_person_missed)

        # Calculate location-specific metrics for test set
        testset_location_correct = sum(count for entity, count in self.testset_results['correct_entities'].items() if '(location)' in entity)
        testset_location_additional = sum(count for entity, count in self.testset_results['additional_entities'].items() if '(location)' in entity)
        testset_location_missed = sum(count for entity, count in self.testset_results['missed_entities'].items() if '(location)' in entity)
        testset_location_metrics = self.calculate_metrics(testset_location_correct, testset_location_additional, testset_location_missed)

        # Test set level results
        print("\nTEST SET LEVEL RESULTS:")
        print("-" * 40)
        print("Exact Mismatches:")
        print(f"  Total: {self.testset_results['exact_mismatches']['total']}")
        print(f"  Person: {self.testset_results['exact_mismatches']['person']}")
        print(f"  Location: {self.testset_results['exact_mismatches']['location']}")

        print("\nFlexible Mismatches:")
        print(f"  Total: {self.testset_results['flexible_mismatches']['total']}")
        print(f"  Person: {self.testset_results['flexible_mismatches']['person']}")
        print(f"  Location: {self.testset_results['flexible_mismatches']['location']}")

        print("\nEntity Counts:")
        print(f"  Correct: {testset_correct}, Additional: {testset_additional}, Missed: {testset_missed}")

        print("\nOverall Metrics:")
        print(f"  Precision: {testset_metrics['precision']:.3f}")
        print(f"  Recall: {testset_metrics['recall']:.3f}")
        print(f"  F1-Score: {testset_metrics['f1']:.3f}")

        print("\nPerson Metrics:")
        print(f"  Precision: {testset_person_metrics['precision']:.3f}")
        print(f"  Recall: {testset_person_metrics['recall']:.3f}")
        print(f"  F1-Score: {testset_person_metrics['f1']:.3f}")

        print("\nLocation Metrics:")
        print(f"  Precision: {testset_location_metrics['precision']:.3f}")
        print(f"  Recall: {testset_location_metrics['recall']:.3f}")
        print(f"  F1-Score: {testset_location_metrics['f1']:.3f}")

        # Top entities
        print("\n")
        print("TOP ENTITIES:")
        print("-" * 40)
        top_additional_entities = dict(Counter(self.testset_results['additional_entities']).most_common(200))
        top_missed_entities = dict(Counter(self.testset_results['missed_entities']).most_common(200))
        top_correct_entities = dict(Counter(self.testset_results['correct_entities']).most_common(200))

        print("Top Additional Entities:")
        for entity in top_additional_entities:
            print(f"  {entity}: {top_additional_entities[entity]}")

        print("\nTop Missed Entities:")
        for entity in top_missed_entities:
            print(f"  {entity}: {top_missed_entities[entity]}")

        print("\nTop Correct Entities:")
        for entity in top_correct_entities:
            print(f"  {entity}: {top_correct_entities[entity]}")

        # File level summary
        print(f"\nFILE LEVEL SUMMARY:")
        print("-" * 40)
        print(f"Number of files processed: {len(self.file_results)}")

        # Show metrics for a few files as examples
        print("\nSample File Metrics:")
        for i, (filename, file_result) in enumerate(list(self.file_results.items())[:3]):
            file_correct = sum(file_result['correct_entities'].values())
            file_additional = sum(file_result['additional_entities'].values())
            file_missed = sum(file_result['missed_entities'].values())
            file_metrics = self.calculate_metrics(file_correct, file_additional, file_missed)
            print(f"  {filename}: P={file_metrics['precision']:.3f}, R={file_metrics['recall']:.3f}, F1={file_metrics['f1']:.3f}")

        # Sentence level summary
        print(f"\nSENTENCE LEVEL SUMMARY:")
        print("-" * 40)
        print(f"Number of sentences processed: {len(self.sentence_results)}")

        print("\nTo view colored sentences, use:")
        print("  - evaluator.print_colored_sentences() for console display")
        print("  - evaluator.display_colored_sentences_html() for HTML display")
        print("  - evaluator.save_colored_sentences_html('output.html') to save HTML file")

    def print_colored_sentences(self, max_sentences=10):
        """Print colored sentences for console inspection (without colors)."""
        print(f"\nSAMPLE SENTENCES WITH ENTITY ANNOTATIONS (first {max_sentences}):")
        print("-" * 60)

        count = 0
        for sentence_id, result in self.sentence_results.items():
            if count >= max_sentences:
                break
            print(f"\n{sentence_id}:")
            print("Raw HTML:", result['colored_sentence'])

            # Also show entity analysis for context
            analysis = result['entity_analysis']
            if analysis['correct']:
                print("Correct entities:", analysis['correct'])
            if analysis['missed']:
                print("Missed entities:", analysis['missed'])
            if analysis['additional']:
                print("Additional entities:", analysis['additional'])
            count += 1

    def display_colored_sentences_html(self, max_sentences=10):
        """Generate HTML content to display colored sentences."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>NER Evaluation Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .sentence {{ margin: 15px 0; padding: 10px; border: 1px solid #ddd; }}
        .sentence-id {{ font-weight: bold; margin-bottom: 5px; }}
        .correct {{ background-color: #90EE90; padding: 2px; border-radius: 3px; }}
        .label-person {{ background-color: #000080; color: white; padding: 2px; border-radius: 3px; }}
        .label-location {{ background-color: #87CEEB; padding: 2px; border-radius: 3px; }}
        .extra-person {{ background-color: #FF8C00; padding: 2px; border-radius: 3px; }}
        .extra-location {{ background-color: #FFD700; padding: 2px; border-radius: 3px; }}
        .legend {{ background-color: #f5f5f5; padding: 10px; margin-bottom: 20px; border-radius: 5px; }}
        .legend-item {{ display: inline-block; margin: 5px 10px; }}
    </style>
</head>
<body>
    <h1>NER Evaluation Results - Colored Sentences</h1>
    
    <div class="legend">
        <h3>Legend:</h3>
        <div class="legend-item"><span class="correct">Correct</span> - Correctly identified entities</div>
        <div class="legend-item"><span class="label-person">Missed Person</span> - Person entities missed by model</div>
        <div class="legend-item"><span class="label-location">Missed Location</span> - Location entities missed by model</div>
        <div class="legend-item"><span class="extra-person">Extra Person</span> - Person entities incorrectly identified</div>
        <div class="legend-item"><span class="extra-location">Extra Location</span> - Location entities incorrectly identified</div>
    </div>
    
    <h2>Sample Sentences (first {max_sentences}):</h2>
"""

        count = 0
        for sentence_id, result in self.sentence_results.items():
            if count >= max_sentences:
                break

            html_content += f"""
    <div class="sentence">
        <div class="sentence-id">{sentence_id}:</div>
        <div>{result['colored_sentence']}</div>
    </div>
"""
            count += 1

        html_content += """
</body>
</html>
"""
        return html_content

    def save_colored_sentences_html(self, filename="ner_evaluation_results.html", max_sentences=50):
        """Save colored sentences to an HTML file."""
        html_content = self.display_colored_sentences_html(max_sentences)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Colored sentences saved to {filename}")
        print(f"Open the file in a web browser to view the colored entities.")

    def get_file_metrics(self, filename=None):
        """Get detailed metrics for a specific file or all files."""
        if filename:
            if filename not in self.file_results:
                print(f"File {filename} not found in results.")
                return None

            file_result = self.file_results[filename]
            correct = sum(file_result['correct_entities'].values())
            additional = sum(file_result['additional_entities'].values())
            missed = sum(file_result['missed_entities'].values())
            metrics = self.calculate_metrics(correct, additional, missed)

            return {
                'filename': filename,
                'counts': {'correct': correct, 'additional': additional, 'missed': missed},
                'metrics': metrics,
                'mismatches': file_result['exact_mismatches']
            }
        else:
            # Return metrics for all files
            all_file_metrics = {}
            for fname, file_result in self.file_results.items():
                correct = sum(file_result['correct_entities'].values())
                additional = sum(file_result['additional_entities'].values())
                missed = sum(file_result['missed_entities'].values())
                metrics = self.calculate_metrics(correct, additional, missed)

                all_file_metrics[fname] = {
                    'counts': {'correct': correct, 'additional': additional, 'missed': missed},
                    'metrics': metrics,
                    'mismatches': file_result['exact_mismatches']
                }

            return all_file_metrics

    def print_detailed_file_metrics(self):
        """Print detailed metrics for each file."""
        print("\nDETAILED FILE METRICS:")
        print("=" * 80)

        for filename, file_result in self.file_results.items():
            correct = sum(file_result['correct_entities'].values())
            additional = sum(file_result['additional_entities'].values())
            missed = sum(file_result['missed_entities'].values())
            metrics = self.calculate_metrics(correct, additional, missed)

            print(f"\n{filename}:")
            print(f"  Entities - Correct: {correct}, Additional: {additional}, Missed: {missed}")
            print(f"  Metrics - Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
            print(f"  Mismatches - Total: {file_result['exact_mismatches']['total']}, Person: {file_result['exact_mismatches']['person']}, Location: {file_result['exact_mismatches']['location']}")


if __name__ == "__main__":
    # Usage examples:
    evaluator = ResultComparer('predicted_entities_final.json')
    sentence_results, file_results, testset_results = evaluator.evaluate()

    # View colored sentences in different ways:
    #evaluator.print_colored_sentences(5)  # Console display
    evaluator.save_colored_sentences_html("results.html", 10000)  # Save to HTML file
    html_content = evaluator.display_colored_sentences_html(10)  # Get HTML string

    # Get detailed metrics:
    #evaluator.print_detailed_file_metrics()  # All files
    file_metric = evaluator.get_file_metrics("1.xml")  # Specific file
    all_metrics = evaluator.get_file_metrics()  # All files as dict
