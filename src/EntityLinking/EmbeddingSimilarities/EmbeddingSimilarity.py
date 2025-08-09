import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class EmbeddingSimilarity:
    def __init__(self, embedding_model, tokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.embedding_model = embedding_model.to(self.device)
        self.tokenizer = tokenizer

    def compute_embedding(self, text):
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Run model
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)

        # Average pooling over tokens (ignores padding)
        embedding = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])

        # Return a 1D tensor
        return embedding.squeeze(0)

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def compute_similarity(self, text1, text2):
        embedding1 = self.compute_embedding(text1)
        embedding2 = self.compute_embedding(text2)

        # Compute cosine similarity
        cosine_similarity = (embedding1 @ embedding2.T).item()

        return cosine_similarity

    def compute_dictionary_embeddings(self, dictionary):
        embeddings = {}
        for key, value in dictionary.items():
            embeddings[key] = self.compute_embedding(value)
        return embeddings

    def find_most_similar_in_dict(self, dictionary, text):
        text_embedding = self.compute_embedding(text)
        max_similarity = float('-inf')
        most_similar_key = None

        for key, value in dictionary.items():
            value_embedding = self.compute_embedding(value)
            similarity = (text_embedding @ value_embedding.T).item()
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_key = key

        return most_similar_key, max_similarity


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    word = "Turicum"

    with open('places_simple.json', 'r', encoding='utf-8') as file:
        localities = json.load(file)

    embedding_similarity = EmbeddingSimilarity(model, tokenizer)
    most_similar_key, similarity = embedding_similarity.find_most_similar_in_dict(localities, word)
    print(f"Most similar key: {localities[most_similar_key]}, Similarity: {similarity}")


