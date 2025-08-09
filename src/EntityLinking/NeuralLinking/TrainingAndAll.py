import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm

# Your existing embedding class
class EmbeddingSimilarity:
    def __init__(self, embedding_model, tokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = embedding_model.to(self.device)
        self.tokenizer = tokenizer

    def compute_embedding(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)

        embedding = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        return embedding.squeeze(0)

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class EntityDatasetBuilder:
    def __init__(self, embedding_similarity, entity_type='person', embeddings_cache_dir='embeddings_cache'):
        """
        Initialize dataset builder for a specific entity type.

        Args:
            embedding_similarity: EmbeddingSimilarity instance
            entity_type: 'person' or 'place'
            embeddings_cache_dir: Directory to save/load embeddings cache
        """
        self.embedding_similarity = embedding_similarity
        self.entity_type = entity_type
        self.embeddings_cache_dir = embeddings_cache_dir
        self.ref_to_id = {}      # Maps reference IDs (p1553, l587) to class indices
        self.id_to_ref = {}      # Maps class indices to reference IDs
        self.embeddings = {}     # Stores precomputed embeddings
        self.training_data = []  # List of (embedding, class_id) pairs

        # Create cache directory if it doesn't exist
        os.makedirs(embeddings_cache_dir, exist_ok=True)

        # Load existing embeddings cache if available
        self._load_embeddings_cache()

    def _get_cache_filepath(self):
        """Get the filepath for the embeddings cache."""
        return os.path.join(self.embeddings_cache_dir, f'{self.entity_type}_embeddings.pkl')

    def _load_embeddings_cache(self):
        """Load embeddings from cache file if it exists."""
        cache_file = self._get_cache_filepath()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.embeddings = cached_data.get('embeddings', {})
                    print(f"Loaded {len(self.embeddings)} cached {self.entity_type} embeddings from {cache_file}")
            except Exception as e:
                print(f"Error loading embeddings cache: {e}")
                self.embeddings = {}
        else:
            print(f"No existing embeddings cache found for {self.entity_type}")

    def _save_embeddings_cache(self):
        """Save embeddings to cache file."""
        cache_file = self._get_cache_filepath()
        try:
            cache_data = {
                'embeddings': self.embeddings,
                'entity_type': self.entity_type,
                'embedding_model': 'intfloat/multilingual-e5-large'
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved {len(self.embeddings)} {self.entity_type} embeddings to {cache_file}")
        except Exception as e:
            print(f"Error saving embeddings cache: {e}")

    def load_json_files(self, json_file_paths: List[str]):
        """Load and process multiple JSON files for the specific entity type."""
        print(f"Loading JSON files for {self.entity_type} entities...")

        # Track new entities that need embeddings
        new_entities = set()

        for file_path in tqdm(json_file_paths, desc="Processing files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for doc_id, doc_content in data.items():
                # Process all sections (summary, body, footnotes)
                for section_name, section_content in doc_content.items():
                    if isinstance(section_content, dict):
                        # Process the appropriate entity type
                        entity_key = 'persNames' if self.entity_type == 'person' else 'placeNames'
                        if entity_key in section_content:
                            entities_to_add = self._collect_entities(section_content[entity_key])
                            new_entities.update(entities_to_add)

        # Compute embeddings for new entities
        if new_entities:
            print(f"Computing embeddings for {len(new_entities)} new {self.entity_type} entities...")
            self._compute_new_embeddings(new_entities)
            # Save updated cache
            self._save_embeddings_cache()
        else:
            print(f"All {self.entity_type} entities already have cached embeddings")

        # Now process all files again to create training data
        print(f"Creating training data for {self.entity_type} entities...")
        for file_path in tqdm(json_file_paths, desc="Creating training data"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for doc_id, doc_content in data.items():
                for section_name, section_content in doc_content.items():
                    if isinstance(section_content, dict):
                        entity_key = 'persNames' if self.entity_type == 'person' else 'placeNames'
                        if entity_key in section_content:
                            self._process_entities(section_content[entity_key])

    def _collect_entities(self, entities_dict: Dict) -> set:
        """Collect entity strings that need embeddings computed."""
        entities_needing_embeddings = set()
        for entity_string in entities_dict.keys():
            if entity_string not in self.embeddings:
                entities_needing_embeddings.add(entity_string)
        return entities_needing_embeddings

    def _compute_new_embeddings(self, entity_strings: set):
        """Compute embeddings for new entity strings."""
        for entity_string in tqdm(entity_strings, desc=f"Computing {self.entity_type} embeddings"):
            embedding = self.embedding_similarity.compute_embedding(entity_string)
            self.embeddings[entity_string] = embedding

    def _process_entities(self, entities_dict: Dict):
        """Process entities from a section and add them to the dataset."""
        for entity_string, entity_info in entities_dict.items():
            ref_id = entity_info['ref']

            # Create a class ID for this reference if it doesn't exist
            if ref_id not in self.ref_to_id:
                class_id = len(self.ref_to_id)
                self.ref_to_id[ref_id] = class_id
                self.id_to_ref[class_id] = ref_id

            class_id = self.ref_to_id[ref_id]

            # Get embedding from cache (should already be computed)
            if entity_string in self.embeddings:
                embedding = self.embeddings[entity_string]
                # Add to training data (avoid duplicates)
                self.training_data.append((embedding, class_id))
            else:
                print(f"Warning: Missing embedding for {entity_string}")

    def validate_dataset(self):
        """Validate that the dataset is consistent."""
        if not self.training_data:
            return False, "No training data available"

        # Check class indices are in valid range
        max_class_id = max(label for _, label in self.training_data)
        min_class_id = min(label for _, label in self.training_data)
        expected_classes = len(self.ref_to_id)

        if max_class_id >= expected_classes:
            return False, f"Class index {max_class_id} exceeds number of classes {expected_classes}"

        if min_class_id < 0:
            return False, f"Negative class index found: {min_class_id}"

        # Check for missing class indices
        present_classes = set(label for _, label in self.training_data)
        expected_classes_set = set(range(expected_classes))
        missing_classes = expected_classes_set - present_classes

        if missing_classes:
            print(f"Warning: Missing training data for classes: {missing_classes}")
            # Map missing classes to their reference IDs
            missing_refs = [self.id_to_ref[class_id] for class_id in missing_classes]
            print(f"Missing reference IDs: {missing_refs}")

        return True, f"Dataset valid with {len(self.training_data)} samples, {expected_classes} classes"

    def get_dataset_info(self):
        """Return information about the created dataset."""
        return {
            'entity_type': self.entity_type,
            'num_entity_strings': len(self.embeddings),
            'num_unique_refs': len(self.ref_to_id),
            'embedding_dim': self.training_data[0][0].shape[0] if self.training_data else 0,
            'reference_ids': list(self.ref_to_id.keys()),
            'cached_embeddings': len(self.embeddings),
            'training_samples': len(self.training_data)
        }

    def create_pytorch_dataset(self):
        """Create a PyTorch dataset from the processed data."""
        return EntityLinkingDataset(self.training_data)

    def clear_embeddings_cache(self):
        """Clear the embeddings cache file."""
        cache_file = self._get_cache_filepath()
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Cleared embeddings cache: {cache_file}")
        self.embeddings = {}

    def get_cache_info(self):
        """Get information about the embeddings cache."""
        cache_file = self._get_cache_filepath()
        cache_exists = os.path.exists(cache_file)
        cache_size = 0

        if cache_exists:
            cache_size = os.path.getsize(cache_file) / (1024 * 1024)  # Size in MB

        return {
            'cache_file': cache_file,
            'cache_exists': cache_exists,
            'cache_size_mb': round(cache_size, 2),
            'cached_entities': len(self.embeddings)
        }


class EntityLinkingDataset(Dataset):
    def __init__(self, training_data: List[Tuple[Tensor, int]]):
        self.embeddings = torch.stack([item[0] for item in training_data])
        self.labels = torch.tensor([item[1] for item in training_data], dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class EntityLinkingNetwork(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super(EntityLinkingNetwork, self).__init__()

        # 8-layer architecture with progressive dimension reduction
        layer_dims = [
            embedding_dim,    # Input layer (1024 for multilingual-e5-large)
            hidden_dim,       # Layer 1 (512)
            hidden_dim // 2,  # Layer 2 (256)
            hidden_dim // 2,  # Layer 3 (256)
            hidden_dim // 4,  # Layer 4 (128)
            hidden_dim // 4,  # Layer 5 (128)
            hidden_dim // 8,  # Layer 6 (64)
            hidden_dim // 8,  # Layer 7 (64)
            num_classes       # Output layer
        ]

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            # Add ReLU and Dropout for all layers except the last one
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EntityLinkingTrainer:
    def __init__(self, model, dataset_builder, batch_size=32, learning_rate=0.001):
        self.model = model
        self.dataset_builder = dataset_builder
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Validate dataset before proceeding
        is_valid, message = self.dataset_builder.validate_dataset()
        if not is_valid:
            raise ValueError(f"Invalid dataset: {message}")
        print(f"Dataset validation: {message}")

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Create dataset and dataloader
        self.dataset = self.dataset_builder.create_pytorch_dataset()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

        # Additional validation
        self._validate_dataloader()

    def _validate_dataloader(self):
        """Validate the dataloader and check for potential issues."""
        try:
            # Get one batch to test
            sample_batch = next(iter(self.dataloader))
            embeddings, labels = sample_batch

            print(f"Batch validation:")
            print(f"  Embeddings shape: {embeddings.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels range: {labels.min().item()} to {labels.max().item()}")
            print(f"  Expected classes: {len(self.dataset_builder.ref_to_id)}")

            # Check if labels are within valid range
            max_label = labels.max().item()
            num_classes = len(self.dataset_builder.ref_to_id)

            if max_label >= num_classes:
                raise ValueError(f"Label {max_label} exceeds number of classes {num_classes}")

        except Exception as e:
            print(f"Dataloader validation error: {e}")
            raise

    def train(self, num_epochs: int):
        """Train the entity linking model with error handling."""
        print(f"Training {self.dataset_builder.entity_type} model on {self.device}")
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Number of classes (unique refs): {len(self.dataset_builder.ref_to_id)}")

        # Additional safety check
        if len(self.dataset_builder.ref_to_id) == 0:
            raise ValueError("No classes found in dataset")

        self.model.train()
        train_losses = []

        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                correct_predictions = 0
                total_predictions = 0

                progress_bar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

                for batch_idx, (batch_embeddings, batch_labels) in enumerate(progress_bar):
                    try:
                        # Move to device
                        batch_embeddings = batch_embeddings.to(self.device)
                        batch_labels = batch_labels.to(self.device)

                        # Additional safety check
                        if batch_labels.max() >= len(self.dataset_builder.ref_to_id):
                            print(f"Error in batch {batch_idx}: Label {batch_labels.max().item()} >= num_classes {len(self.dataset_builder.ref_to_id)}")
                            continue

                        # Zero gradients
                        self.optimizer.zero_grad()

                        # Forward pass
                        outputs = self.model(batch_embeddings)
                        loss = self.criterion(outputs, batch_labels)

                        # Check for NaN loss
                        if torch.isnan(loss):
                            print(f"NaN loss detected in batch {batch_idx}")
                            continue

                        # Backward pass
                        loss.backward()

                        # Gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                        self.optimizer.step()

                        # Statistics
                        epoch_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_predictions += batch_labels.size(0)
                        correct_predictions += (predicted == batch_labels).sum().item()

                        # Update progress bar
                        progress_bar.set_postfix({
                            'Loss': f'{loss.item():.4f}',
                            'Acc': f'{100.0 * correct_predictions / total_predictions:.2f}%'
                        })

                    except RuntimeError as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        if "CUDA" in str(e):
                            print("CUDA error detected. Trying to continue...")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise

                # Calculate epoch statistics
                if len(self.dataloader) > 0:
                    avg_loss = epoch_loss / len(self.dataloader)
                    accuracy = 100.0 * correct_predictions / total_predictions if total_predictions > 0 else 0
                    train_losses.append(avg_loss)

                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
                else:
                    print(f'Epoch [{epoch+1}/{num_epochs}], No valid batches processed')

        except Exception as e:
            print(f"Training error: {e}")
            # Clear CUDA cache if CUDA error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        return train_losses

    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ref_to_id': self.dataset_builder.ref_to_id,
            'id_to_ref': self.dataset_builder.id_to_ref,
            'entity_type': self.dataset_builder.entity_type,
            'embeddings': self.dataset_builder.embeddings,  # Save for reference
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.dataset_builder.ref_to_id = checkpoint['ref_to_id']
        self.dataset_builder.id_to_ref = checkpoint['id_to_ref']
        self.dataset_builder.entity_type = checkpoint['entity_type']
        print(f"Model loaded from {filepath}")


class EntityLinker:
    def __init__(self, model_path: str, embedding_similarity: EmbeddingSimilarity):
        """
        Initialize the EntityLinker with a trained model.

        Args:
            model_path: Path to the saved model file
            embedding_similarity: EmbeddingSimilarity instance for computing embeddings
        """
        self.embedding_similarity = embedding_similarity
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load the trained model and mappings
        self.model = None
        self.ref_to_id = {}      # Maps reference IDs to class indices
        self.id_to_ref = {}      # Maps class indices to reference IDs
        self.entity_type = None
        self.training_embeddings = {}

        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the trained model and associated mappings."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Load mappings
            self.ref_to_id = checkpoint['ref_to_id']
            self.id_to_ref = checkpoint['id_to_ref']
            self.entity_type = checkpoint['entity_type']
            self.training_embeddings = checkpoint.get('embeddings', {})

            # Reconstruct the model
            embedding_dim = 1024  # multilingual-e5-large embedding dimension
            num_classes = len(self.ref_to_id)

            self.model = EntityLinkingNetwork(
                embedding_dim=embedding_dim,
                hidden_dim=512,
                num_classes=num_classes,
                dropout_rate=0.3
            )

            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            print(f"{self.entity_type.capitalize()} model loaded successfully from {model_path}")
            print(f"Model can predict {num_classes} different {self.entity_type} reference IDs")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_entity_link(self, entity_string: str, top_k: int = 1, return_probabilities: bool = False):
        """
        Predict the reference ID for a given entity string.

        Args:
            entity_string: The entity string to classify
            top_k: Number of top predictions to return
            return_probabilities: Whether to return probability scores

        Returns:
            If top_k=1 and return_probabilities=False: reference ID (e.g., 'p1553')
            If top_k=1 and return_probabilities=True: (reference_id, probability)
            If top_k>1: List of tuples [(reference_id, probability), ...]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")

        try:
            # Compute embedding for the input string
            embedding = self.embedding_similarity.compute_embedding(entity_string)
            embedding = embedding.unsqueeze(0).to(self.device)  # Add batch dimension

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(embedding)
                probabilities = torch.softmax(outputs, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)))

            # Convert to results
            results = []
            for i in range(top_probs.size(1)):
                class_id = top_indices[0][i].item()
                probability = top_probs[0][i].item()
                reference_id = self.id_to_ref[class_id]
                results.append((reference_id, probability))

            # Return based on parameters
            if top_k == 1:
                if return_probabilities:
                    return results[0]
                else:
                    return results[0][0]
            else:
                return results

        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def predict_with_similarity(self, entity_string: str, similarity_threshold: float = 0.5):
        """
        Predict entity link and also show the most similar training entities.

        Args:
            entity_string: The entity string to classify
            similarity_threshold: Minimum similarity to show training entities

        Returns:
            Dictionary with prediction and similar training entities
        """
        # Get the main prediction
        prediction, probability = self.predict_entity_link(
            entity_string,
            return_probabilities=True
        )

        # Compute embedding for input
        input_embedding = self.embedding_similarity.compute_embedding(entity_string)

        # Find similar entities in training data
        similar_entities = []
        for train_entity, train_embedding in self.training_embeddings.items():
            if train_entity != entity_string:  # Skip exact matches
                # Compute cosine similarity
                similarity = torch.cosine_similarity(
                    input_embedding.unsqueeze(0),
                    train_embedding.unsqueeze(0)
                ).item()

                if similarity >= similarity_threshold:
                    similar_entities.append((train_entity, similarity))

        # Sort by similarity
        similar_entities.sort(key=lambda x: x[1], reverse=True)

        return {
            'prediction': {
                'reference_id': prediction,
                'probability': probability
            },
            'input_string': entity_string,
            'entity_type': self.entity_type,
            'similar_training_entities': similar_entities[:5]  # Top 5 similar
        }

    def get_all_entities_for_ref(self, reference_id: str):
        """
        Get all entity strings that map to a specific reference ID from training data.

        Args:
            reference_id: The reference ID (e.g., 'p1553')

        Returns:
            List of entity strings that mapped to this reference during training
        """
        if reference_id not in self.ref_to_id:
            return []

        # This would require storing the mapping during training
        # For now, we can't retrieve this without additional data structure
        return [f"Training data for {reference_id} not stored in model"]

    def get_model_info(self):
        """Get information about the loaded model."""
        if self.model is None:
            return "No model loaded"

        return {
            'entity_type': self.entity_type,
            'total_reference_ids': len(self.ref_to_id),
            'reference_ids': list(self.ref_to_id.keys()),
            'device': self.device,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }


def train_both_models(json_files: List[str], embeddings_cache_dir='embeddings_cache'):
    """Train separate models for persons and places with embeddings caching."""
    # Initialize the embedding model
    print("Loading embedding model...")
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    embedding_similarity = EmbeddingSimilarity(model, tokenizer)

    # Training parameters
    training_params = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'hidden_dim': 512
    }

    # Train person model
    print("\n" + "="*50)
    print("TRAINING PERSON MODEL")
    print("="*50)

    person_dataset_builder = EntityDatasetBuilder(
        embedding_similarity,
        entity_type='person',
        embeddings_cache_dir=embeddings_cache_dir
    )

    # Show cache info
    cache_info = person_dataset_builder.get_cache_info()
    print(f"Person embeddings cache info: {cache_info}")

    person_dataset_builder.load_json_files(json_files)

    person_info = person_dataset_builder.get_dataset_info()
    print(f"Person dataset info: {person_info}")

    if person_info['num_entity_strings'] > 0:
        person_network = EntityLinkingNetwork(
            embedding_dim=person_info['embedding_dim'],
            hidden_dim=training_params['hidden_dim'],
            num_classes=person_info['num_unique_refs'],
            dropout_rate=0.3
        )

        person_trainer = EntityLinkingTrainer(
            model=person_network,
            dataset_builder=person_dataset_builder,
            batch_size=training_params['batch_size'],
            learning_rate=training_params['learning_rate']
        )

        person_losses = person_trainer.train(training_params['num_epochs'])
        person_trainer.save_model('person_entity_linking_model.pth')
        print("Person model training completed!")

    # Train place model
    print("\n" + "="*50)
    print("TRAINING PLACE MODEL")
    print("="*50)

    place_dataset_builder = EntityDatasetBuilder(
        embedding_similarity,
        entity_type='place',
        embeddings_cache_dir=embeddings_cache_dir
    )

    # Show cache info
    cache_info = place_dataset_builder.get_cache_info()
    print(f"Place embeddings cache info: {cache_info}")

    place_dataset_builder.load_json_files(json_files)

    place_info = place_dataset_builder.get_dataset_info()
    print(f"Place dataset info: {place_info}")

    if place_info['num_entity_strings'] > 0:
        place_network = EntityLinkingNetwork(
            embedding_dim=place_info['embedding_dim'],
            hidden_dim=training_params['hidden_dim'],
            num_classes=place_info['num_unique_refs'],
            dropout_rate=0.3
        )

        place_trainer = EntityLinkingTrainer(
            model=place_network,
            dataset_builder=place_dataset_builder,
            batch_size=training_params['batch_size'],
            learning_rate=training_params['learning_rate']
        )

        place_losses = place_trainer.train(training_params['num_epochs'])
        place_trainer.save_model('place_entity_linking_model.pth')
        print("Place model training completed!")


def clear_all_caches(embeddings_cache_dir='embeddings_cache'):
    """Clear all embeddings caches."""
    print("Clearing all embeddings caches...")

    # Clear person cache
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    embedding_similarity = EmbeddingSimilarity(model, tokenizer)

    person_builder = EntityDatasetBuilder(
        embedding_similarity,
        entity_type='person',
        embeddings_cache_dir=embeddings_cache_dir
    )
    person_builder.clear_embeddings_cache()

    place_builder = EntityDatasetBuilder(
        embedding_similarity,
        entity_type='place',
        embeddings_cache_dir=embeddings_cache_dir
    )
    place_builder.clear_embeddings_cache()

    print("All caches cleared!")


def show_cache_status(embeddings_cache_dir='embeddings_cache'):
    """Show the status of all embeddings caches."""
    print("Embeddings Cache Status:")
    print("="*30)

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    embedding_similarity = EmbeddingSimilarity(model, tokenizer)

    for entity_type in ['person', 'place']:
        builder = EntityDatasetBuilder(
            embedding_similarity,
            entity_type=entity_type,
            embeddings_cache_dir=embeddings_cache_dir
        )
        cache_info = builder.get_cache_info()
        print(f"\n{entity_type.capitalize()} Cache:")
        for key, value in cache_info.items():
            print(f"  {key}: {value}")


def demo_prediction():
    """Demonstrate how to use both EntityLinkers for predictions."""
    # Initialize the embedding model (same as used for training)
    print("Loading embedding model for predictions...")
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    embedding_similarity = EmbeddingSimilarity(model, tokenizer)

    # Load both trained models
    person_linker = EntityLinker('person_entity_linking_model.pth', embedding_similarity)
    place_linker = EntityLinker('place_entity_linking_model.pth', embedding_similarity)

    # Example predictions for persons
    person_entities = [
        "Hochholzer",
        "Rudolf Gwalther",
        "Gvaltherum",  # Latin form
        "Johannes Haller",
        "Hallerum"    # Latin form
    ]

    print("\n=== PERSON Entity Linking Predictions ===")
    for entity in person_entities:
        print(f"\nInput: '{entity}'")

        # Simple prediction
        predicted_ref = person_linker.predict_entity_link(entity)
        print(f"Predicted reference ID: {predicted_ref}")

        # Prediction with probability
        predicted_ref, probability = person_linker.predict_entity_link(
            entity, return_probabilities=True
        )
        print(f"Prediction with confidence: {predicted_ref} ({probability:.3f})")

        # Top-3 predictions
        top_predictions = person_linker.predict_entity_link(entity, top_k=3)
        print("Top 3 predictions:")
        for i, (ref, prob) in enumerate(top_predictions, 1):
            print(f"  {i}. {ref} ({prob:.3f})")

    # Example predictions for places
    place_entities = [
        "Zürich",
        "Tiguri",      # Latin form of Zürich
        "Tigurinae",   # Another Latin form
        "Aarau",
        "Aarovię"      # Latin form of Aarau
    ]

    print("\n=== PLACE Entity Linking Predictions ===")
    for entity in place_entities:
        print(f"\nInput: '{entity}'")

        # Simple prediction
        predicted_ref = place_linker.predict_entity_link(entity)
        print(f"Predicted reference ID: {predicted_ref}")

        # Prediction with probability and similarity
        detailed_result = place_linker.predict_with_similarity(entity, similarity_threshold=0.7)
        print(f"Prediction: {detailed_result['prediction']['reference_id']} "
              f"(confidence: {detailed_result['prediction']['probability']:.3f})")

        print("Similar training entities:")
        for train_entity, similarity in detailed_result['similar_training_entities']:
            print(f"  '{train_entity}' (similarity: {similarity:.3f})")

    # Show model information
    print(f"\n=== Model Information ===")
    person_info = person_linker.get_model_info()
    place_info = place_linker.get_model_info()

    print("Person Model:")
    for key, value in person_info.items():
        print(f"  {key}: {value}")

    print("Place Model:")
    for key, value in place_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Replace with your actual JSON file paths
    json_files = glob.glob('C:/Users/MartinFaehnrich/Documents/BullingerDigi/src/PreProcessing/Entities/*.json')

    # Uncomment the function you want to run:

    # For training both models:
    train_both_models(json_files)

    # For prediction demo (after training):
    demo_prediction()