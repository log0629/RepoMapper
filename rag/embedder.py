from typing import List, Any
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self.model = SentenceTransformer(model)

    def embed_text(self, text: str) -> List[float]:
        """
        Generates an embedding for a single text string.
        """
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of text strings.
        """
        if not texts:
            return []
            
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
