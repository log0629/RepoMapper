from typing import List, Any

class Embedder:
    def __init__(self, client: Any, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        """
        Generates an embedding for a single text string.
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of text strings.
        """
        if not texts:
            return []
            
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]
