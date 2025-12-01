import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock sentence_transformers before importing rag modules
sys.modules['sentence_transformers'] = MagicMock()
# Mock qdrant_client before importing rag modules
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.http'] = MagicMock()
sys.modules['qdrant_client.http.models'] = MagicMock()

from rag.indexer import RepoIndexer, COLLECTION_REPOS, COLLECTION_BLOCKS

class TestRagIndexer(unittest.TestCase):
    def setUp(self):
        self.mock_qdrant_client = MagicMock()
        with patch('rag.indexer.QdrantClient', return_value=self.mock_qdrant_client):
            self.indexer = RepoIndexer()

    def test_create_collections(self):
        # Mock collection_exists to return False then True
        self.mock_qdrant_client.collection_exists.side_effect = [False, False]
        
        self.indexer.create_collections()
        
        self.assertEqual(self.mock_qdrant_client.create_collection.call_count, 2)
        self.mock_qdrant_client.create_collection.assert_any_call(
            collection_name=COLLECTION_REPOS,
            vectors_config=unittest.mock.ANY
        )
        self.mock_qdrant_client.create_collection.assert_any_call(
            collection_name=COLLECTION_BLOCKS,
            vectors_config=unittest.mock.ANY
        )

    def test_get_last_commit_sha(self):
        # Mock scroll response
        mock_point = MagicMock()
        mock_point.payload = {"commit_sha": "test_sha"}
        self.mock_qdrant_client.scroll.return_value = ([mock_point], None)
        
        sha = self.indexer.get_last_commit_sha("test/repo")
        self.assertEqual(sha, "test_sha")
        
        # Verify scroll call
        self.mock_qdrant_client.scroll.assert_called_once()
        call_args = self.mock_qdrant_client.scroll.call_args
        self.assertEqual(call_args.kwargs['collection_name'], COLLECTION_REPOS)

    def test_index_repository_data(self):
        # Test the core indexing logic (upserting data)
        # We assume the data is already extracted (passed as args)
        
        repo_id = "test/repo"
        commit_sha = "new_sha"
        summary = "test summary"
        em_summary = [0.1, 0.2]
        blocks = [{
            "file_path": "test.py",
            "name": "func1",
            "type": "function",
            "content": "def func1(): pass",
            "rank_score": 1.0,
            "em_content": [0.3, 0.4],
            "start_line": 1,
            "end_line": 2,
            "repo_id": repo_id
        }]
        
        self.indexer.index_repository_data(
            repo_id=repo_id,
            commit_sha=commit_sha,
            summary=summary,
            em_summary=em_summary,
            blocks=blocks
        )
        
        # Verify upserts
        self.assertEqual(self.mock_qdrant_client.upsert.call_count, 2)
        
        # Verify repo upsert
        repo_upsert_call = self.mock_qdrant_client.upsert.call_args_list[0]
        self.assertEqual(repo_upsert_call.kwargs['collection_name'], COLLECTION_REPOS)
        
        # Check that PointStruct was called with correct payload
        # Since PointStruct is mocked, we check the call args of the mock class
        # But wait, we mocked qdrant_client.http.models.PointStruct
        # Let's check the points passed to upsert. They are instances of the mocked PointStruct.
        # It's hard to inspect the mock instance attributes if they weren't set explicitly.
        # Instead, let's verify that PointStruct was instantiated with the correct payload.
        
        # Find the call to PointStruct for repo
        # We expect 1 call for repo and 1 call for block
        # But upsert receives a list of points.
        
        # Let's inspect the mock_qdrant_client.upsert call arguments directly
        # The 'points' argument contains the result of PointStruct(...) calls.
        # Since PointStruct is a mock, these are mock objects.
        
        # Better approach: Mock PointStruct to return a dict or simple object so we can inspect it
        # But we mocked the module.
        
        # Let's just verify that PointStruct was CALLED with the correct arguments.
        # We need to import the mocked PointStruct to check its calls.
        from qdrant_client.http import models
        
        # Filter calls to PointStruct
        # We expect calls with payload containing commit_sha
        found_repo_call = False
        for call in models.PointStruct.call_args_list:
            if 'payload' in call.kwargs and call.kwargs['payload'].get('commit_sha') == commit_sha:
                found_repo_call = True
                break
        
        self.assertTrue(found_repo_call, "PointStruct not called with correct repo payload")

if __name__ == '__main__':
    unittest.main()
