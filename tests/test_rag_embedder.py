import unittest
from unittest.mock import MagicMock, patch
import sys
import numpy as np

# Mock sentence_transformers module
mock_st = MagicMock()
sys.modules["sentence_transformers"] = mock_st

from rag.embedder import Embedder

class TestEmbedder(unittest.TestCase):
    @patch('rag.embedder.SentenceTransformer')
    def setUp(self, mock_st_cls):
        self.mock_model = MagicMock()
        mock_st_cls.return_value = self.mock_model
        self.embedder = Embedder(model="test-model")

    def test_embed_text(self):
        # Mock response (numpy array)
        self.mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        
        # Call method
        embedding = self.embedder.embed_text("test text")
        
        # Verify
        self.mock_model.encode.assert_called_with("test text")
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    def test_embed_batch(self):
        # Mock response
        self.mock_model.encode.return_value = np.array([[0.1], [0.2]])
        
        # Call method
        embeddings = self.embedder.embed_batch(["text1", "text2"])
        
        # Verify
        self.mock_model.encode.assert_called_with(["text1", "text2"])
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1])
        self.assertEqual(embeddings[1], [0.2])

if __name__ == '__main__':
    unittest.main()
