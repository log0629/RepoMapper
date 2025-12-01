import unittest
from unittest.mock import MagicMock
from rag.embedder import Embedder

class TestEmbedder(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.embedder = Embedder(client=self.mock_client)

    def test_embed_text(self):
        # Mock response
        self.mock_client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        
        # Call method
        embedding = self.embedder.embed_text("test text")
        
        # Verify
        self.mock_client.embeddings.create.assert_called_once()
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    def test_embed_batch(self):
        # Mock response
        mock_data = [MagicMock(embedding=[0.1]), MagicMock(embedding=[0.2])]
        self.mock_client.embeddings.create.return_value.data = mock_data
        
        # Call method
        embeddings = self.embedder.embed_batch(["text1", "text2"])
        
        # Verify
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1])
        self.assertEqual(embeddings[1], [0.2])

if __name__ == '__main__':
    unittest.main()
