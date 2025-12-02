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
        
        # Mock scroll for get_stored_block_ids
        self.mock_qdrant_client.scroll.return_value = ([], None)
        
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
        from qdrant_client.http import models
        found_repo_call = False
        for call in models.PointStruct.call_args_list:
            if 'payload' in call.kwargs and call.kwargs['payload'].get('commit_sha') == commit_sha:
                found_repo_call = True
                break
        
        self.assertTrue(found_repo_call, "PointStruct not called with correct repo payload")

    def test_index_repository_smart_diffing(self):
        # Setup:
        # - Stored IDs: [id_keep, id_delete]
        # - Current Blocks: [block_keep, block_add]
        # Expected:
        # - Upsert: [block_keep, block_add] (We upsert all current blocks for now)
        # - Delete: [id_delete]
        
        repo_id = "test/repo"
        commit_sha = "new_sha"
        summary = "summary"
        em_summary = [0.1]
        
        # Mock ID generation to be predictable
        def mock_generate_id(key):
            if "keep" in key: return "id_keep"
            if "add" in key: return "id_add"
            if "delete" in key: return "id_delete"
            return "id_unknown"
            
        self.indexer._generate_id = mock_generate_id
        
        # Mock get_stored_block_ids
        self.indexer.get_stored_block_ids = MagicMock(return_value={"id_keep", "id_delete"})
        
        blocks = [
            {"name": "keep", "file_path": "f1", "start_line": 1, "em_content": [0.1], "content": "c", "type": "t", "rank_score": 1.0, "end_line": 2},
            {"name": "add", "file_path": "f2", "start_line": 1, "em_content": [0.2], "content": "c", "type": "t", "rank_score": 1.0, "end_line": 2}
        ]
        
        self.indexer.index_repository_data(repo_id, commit_sha, summary, em_summary, blocks)
        
        # Verify Upsert (Repo Info + Added Block)
        # Repo info is always upserted
        self.assertEqual(self.mock_qdrant_client.upsert.call_count, 2)
        
        # Verify Block Upsert: Should contain BOTH 'id_keep' and 'id_add'
        # Since points are mocks, we verify the PointStruct calls used to create them
        from qdrant_client.http import models
        
        upserted_ids = set()
        for call in models.PointStruct.call_args_list:
            if 'id' in call.kwargs:
                upserted_ids.add(call.kwargs['id'])
                
        self.assertIn("id_keep", upserted_ids)
        self.assertIn("id_add", upserted_ids)
        
        # Verify Delete: Should contain 'id_delete'
        self.mock_qdrant_client.delete.assert_called_once()
        
        # Verify PointIdsList was called with correct points
        point_ids_list_call = models.PointIdsList.call_args
        self.assertEqual(point_ids_list_call.kwargs['points'], ["id_delete"])

if __name__ == '__main__':
    unittest.main()
