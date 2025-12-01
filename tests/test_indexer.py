import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexer import RepoIndexer, COLLECTION_REPOS, COLLECTION_BLOCKS

class TestIndexer(unittest.TestCase):
    def setUp(self):
        self.mock_qdrant_client = MagicMock()
        with patch('indexer.QdrantClient', return_value=self.mock_qdrant_client):
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

    @patch('indexer.requests.post')
    def test_index_repository_new(self, mock_post):
        # Mock API responses
        # 1. /repomap
        mock_repomap_resp = MagicMock()
        mock_repomap_resp.json.return_value = {
            "repo_map": "map content",
            "repo_id": "test/repo",
            "commit_sha": "new_sha"
        }
        mock_repomap_resp.raise_for_status.return_value = None
        
        # 2. /embed/summary
        mock_summary_resp = MagicMock()
        mock_summary_resp.json.return_value = {
            "summary": "summary text",
            "em_summary": [0.1, 0.2],
            "repo_id": "test/repo"
        }
        
        # 3. /semantic-blocks
        mock_sem_blocks_resp = MagicMock()
        mock_sem_blocks_resp.json.return_value = {
            "blocks": [{"name": "block1"}],
            "commit_sha": "new_sha"
        }
        
        # 4. /embed/blocks
        mock_embed_blocks_resp = MagicMock()
        mock_embed_blocks_resp.json.return_value = {
            "blocks": [{
                "file_path": "test.py",
                "name": "block1",
                "type": "function",
                "content": "def block1(): pass",
                "rank_score": 1.0,
                "em_content": [0.3, 0.4],
                "start_line": 1,
                "end_line": 2
            }]
        }

        mock_post.side_effect = [
            mock_repomap_resp, 
            mock_summary_resp, 
            mock_sem_blocks_resp, 
            mock_embed_blocks_resp
        ]

        # Mock get_last_commit_sha to return None (new repo)
        self.indexer.get_last_commit_sha = MagicMock(return_value=None)
        
        self.indexer.index_repository("/tmp/repo", "test/repo")
        
        # Verify upserts
        self.assertEqual(self.mock_qdrant_client.upsert.call_count, 2)
        
        # Verify repo upsert
        repo_upsert_call = self.mock_qdrant_client.upsert.call_args_list[0]
        self.assertEqual(repo_upsert_call.kwargs['collection_name'], COLLECTION_REPOS)
        self.assertEqual(repo_upsert_call.kwargs['points'][0].payload['commit_sha'], "new_sha")
        
        # Verify blocks upsert
        blocks_upsert_call = self.mock_qdrant_client.upsert.call_args_list[1]
        self.assertEqual(blocks_upsert_call.kwargs['collection_name'], COLLECTION_BLOCKS)
        self.assertEqual(len(blocks_upsert_call.kwargs['points']), 1)

    @patch('indexer.requests.post')
    def test_index_repository_skip_if_unchanged(self, mock_post):
        # Mock API response for /repomap
        mock_repomap_resp = MagicMock()
        mock_repomap_resp.json.return_value = {
            "repo_map": "map content",
            "repo_id": "test/repo",
            "commit_sha": "existing_sha"
        }
        mock_post.return_value = mock_repomap_resp
        
        # Mock get_last_commit_sha to return same SHA
        self.indexer.get_last_commit_sha = MagicMock(return_value="existing_sha")
        
        self.indexer.index_repository("/tmp/repo", "test/repo")
        
        # Verify NO upserts
        self.mock_qdrant_client.upsert.assert_not_called()
        # Verify only 1 API call (checking SHA)
        self.assertEqual(mock_post.call_count, 1)

if __name__ == '__main__':
    unittest.main()
