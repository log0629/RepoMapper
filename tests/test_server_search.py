import unittest
from unittest.mock import MagicMock, patch
import sys
from fastapi.testclient import TestClient

# Mock dependencies before importing server.main
sys.modules['openai'] = MagicMock()
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.http'] = MagicMock()
sys.modules['qdrant_client.http.models'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

from server.main import app

class TestServerSearch(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    @patch('server.main.embedder')
    @patch('server.main.indexer')
    def test_search_repos(self, mock_indexer, mock_embedder):
        # Setup Mocks
        mock_embedder.embed_text.return_value = [0.1, 0.2]
        mock_indexer.search_repositories.return_value = [
            {"repo_id": "repo1", "score": 0.9, "summary": "summary1"}
        ]
        
        # Request
        response = self.client.post("/search/repos", json={"query": "test query"})
        
        # Verify
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['repo_id'], "repo1")
        
        # Verify calls
        mock_embedder.embed_text.assert_called_with("test query")
        mock_indexer.search_repositories.assert_called_with(
            query_vector=[0.1, 0.2],
            limit=5
        )

    @patch('server.main.embedder')
    @patch('server.main.indexer')
    def test_search_code(self, mock_indexer, mock_embedder):
        # Setup Mocks
        mock_embedder.embed_text.return_value = [0.1, 0.2]
        mock_indexer.search_code_blocks.return_value = [
            {
                "repo_id": "repo1", 
                "file_path": "f1", 
                "name": "n1", 
                "content": "c1", 
                "start_line": 1, 
                "score": 0.8
            }
        ]
        
        # Request
        response = self.client.post("/search/code", json={
            "query": "test code",
            "repo_ids": ["repo1"]
        })
        
        # Verify
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['name'], "n1")
        
        # Verify calls
        mock_embedder.embed_text.assert_called_with("test code")
        mock_indexer.search_code_blocks.assert_called_with(
            query_vector=[0.1, 0.2],
            repo_ids=["repo1"],
            limit=10
        )

    @patch('server.main.embedder')
    @patch('server.main.indexer')
    def test_search_unified(self, mock_indexer, mock_embedder):
        # Setup Mocks
        mock_embedder.embed_text.return_value = [0.1, 0.2]
        
        # Mock Repo Search Results
        mock_indexer.search_repositories.return_value = [
            {"repo_id": "repo1", "score": 0.9, "summary": "summary1"}
        ]
        
        # Mock Block Search Results
        mock_indexer.search_code_blocks.return_value = [
            {
                "repo_id": "repo1", 
                "file_path": "f1", 
                "name": "n1", 
                "content": "c1", 
                "start_line": 1, 
                "score": 0.8
            }
        ]
        
        # Request
        response = self.client.post("/search/unified", json={"query": "test unified"})
        
        # Verify
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check Repos
        self.assertEqual(len(data['repositories']), 1)
        self.assertEqual(data['repositories'][0]['repo_id'], "repo1")
        
        # Check Blocks
        self.assertEqual(len(data['blocks']), 1)
        self.assertEqual(data['blocks'][0]['name'], "n1")
        
        # Verify calls
        mock_embedder.embed_text.assert_called_once_with("test unified")
        
        # Verify Repo Search
        mock_indexer.search_repositories.assert_called_with(
            query_vector=[0.1, 0.2],
            limit=5
        )
        
        # Verify Block Search (should be filtered by repo1)
        mock_indexer.search_code_blocks.assert_called_with(
            query_vector=[0.1, 0.2],
            repo_ids=["repo1"],
            limit=10
        )

if __name__ == '__main__':
    unittest.main()
