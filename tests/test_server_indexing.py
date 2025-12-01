import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies before importing server.main
sys.modules['openai'] = MagicMock()
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.http'] = MagicMock()
sys.modules['qdrant_client.http.models'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

from server.main import app

class TestServerIndexing(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('server.main.indexer')
    @patch('server.main.manager')
    def test_index_repository(self, mock_manager, mock_indexer):
        # Mock manager responses
        mock_manager.extract_repo_map.return_value = ("repo_map_content", "new_sha")
        mock_manager.extract_semantic_blocks.return_value = ([{
            "name": "block1", 
            "content": "def block1(): pass",
            "file_path": "test.py",
            "type": "function",
            "start_line": 1,
            "end_line": 2,
            "rank_score": 1.0
        }], "new_sha")
        
        # Mock indexer responses
        mock_indexer.get_last_commit_sha.return_value = "old_sha"
        
        # Mock generator and embedder (used inside embed_summary and embed_blocks)
        with patch('server.main.generator') as mock_gen, \
             patch('server.main.embedder') as mock_emb:
            
            mock_gen.generate_summary.return_value = "summary text"
            mock_emb.embed_text.return_value = [0.1, 0.2]
            mock_emb.embed_batch.return_value = [[0.3, 0.4]]
            
            response = self.client.post("/index", json={
                "root_path": "/tmp/repo",
                "repo_id": "test/repo"
            })
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "indexed")
            self.assertEqual(response.json()["commit_sha"], "new_sha")
            
            # Verify indexer called
            mock_indexer.index_repository_data.assert_called_once()
            call_args = mock_indexer.index_repository_data.call_args
            self.assertEqual(call_args.kwargs['repo_id'], "test/repo")
            self.assertEqual(call_args.kwargs['commit_sha'], "new_sha")
            self.assertEqual(call_args.kwargs['summary'], "summary text")
            self.assertEqual(len(call_args.kwargs['blocks']), 1)

    @patch('server.main.indexer')
    @patch('server.main.manager')
    def test_index_repository_skip(self, mock_manager, mock_indexer):
        # Mock manager responses
        mock_manager.extract_repo_map.return_value = ("repo_map_content", "same_sha")
        
        # Mock indexer responses
        mock_indexer.get_last_commit_sha.return_value = "same_sha"
        
        response = self.client.post("/index", json={
            "root_path": "/tmp/repo",
            "repo_id": "test/repo"
        })
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "skipped")
        self.assertEqual(response.json()["commit_sha"], "same_sha")
        
        # Verify indexer NOT called
        mock_indexer.index_repository_data.assert_not_called()

if __name__ == '__main__':
    unittest.main()
