import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock openai module before importing server.main
mock_openai = MagicMock()
sys.modules["openai"] = mock_openai

# Mock sentence_transformers module
mock_st = MagicMock()
sys.modules["sentence_transformers"] = mock_st

from fastapi.testclient import TestClient
from server.main import app
from server.models import RepoRequest, RepoMapResponse, SemanticBlocksResponse

class TestServer(unittest.TestCase):
    def setUp(self):
        from server.main import manager
        manager.repos = {}
        self.client = TestClient(app)

    @patch('server.manager.find_src_files')
    @patch('server.manager.RepoMap')
    def test_extract_repomap(self, mock_repomap_cls, mock_find_files):
        # Mock find_src_files
        mock_find_files.return_value = ["test.py"]
        
        # Mock RepoMap instance
        mock_instance = MagicMock()
        mock_instance.get_repo_map.return_value = ("Mock Repo Map", None)
        mock_repomap_cls.return_value = mock_instance

        response = self.client.post("/repomap", json={"root_path": "/tmp/test_repo"})
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["repo_map"], "Mock Repo Map")

    @patch('server.manager.find_src_files')
    @patch('server.manager.RepoMap')
    def test_extract_repomap_with_options(self, mock_repomap_cls, mock_find_files):
        # Mock find_src_files
        mock_find_files.return_value = ["other.py"]
        
        # Mock RepoMap instance
        mock_instance = MagicMock()
        mock_instance.get_repo_map.return_value = ("Mock Repo Map", None)
        mock_repomap_cls.return_value = mock_instance

        payload = {
            "root_path": "/tmp/test_repo",
            "token_limit": 2048,
            "chat_files": ["main.py"],
            "verbose": True,
            "exclude_unranked": True
        }
        response = self.client.post("/repomap", json=payload)
        
        self.assertEqual(response.status_code, 200)
        
        # Verify RepoMap init args
        mock_repomap_cls.assert_called_with(
            root="/tmp/test_repo",
            map_tokens=2048,
            token_counter_func=unittest.mock.ANY,
            verbose=True,
            max_context_window=None,
            exclude_unranked=True
        )
        
        # Verify get_repo_map args
        mock_instance.get_repo_map.assert_called_with(
            chat_files=["main.py"],
            other_files=["other.py"],
            mentioned_fnames=None,
            mentioned_idents=None,
            force_refresh=False
        )

    @patch('server.manager.find_src_files')
    @patch('server.manager.RepoMap')
    def test_extract_semantic_blocks(self, mock_repomap_cls, mock_find_files):
        # Mock find_src_files
        mock_find_files.return_value = ["test.py"]

        # Mock RepoMap instance
        mock_instance = MagicMock()
        mock_block = MagicMock()
        mock_block.file_path = "test.py"
        mock_block.name = "test_func"
        mock_block.rank_score = 1.0
        mock_block.content = "def test_func(): pass"
        mock_block.type = "function_definition"
        mock_block.start_line = 1
        mock_block.end_line = 2
        
        # Mock asdict behavior: we need to patch asdict or make mock_block look like a dataclass
        # Easier: patch asdict in manager
        with patch('server.manager.asdict') as mock_asdict:
            mock_asdict.return_value = {
                "file_path": "test.py",
                "type": "function_definition",
                "name": "test_func",
                "start_line": 1,
                "end_line": 2,
                "content": "def test_func(): pass",
                "rank_score": 1.0
            }
            
            mock_instance.get_semantic_blocks.return_value = [mock_block]
            mock_repomap_cls.return_value = mock_instance

            response = self.client.post("/semantic-blocks", json={"root_path": "/tmp/test_repo"})
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["blocks"]), 1)
            self.assertEqual(response.json()["blocks"][0]["name"], "test_func")

    @patch('server.main.generator')
    def test_summarize_repomap(self, mock_generator):
        mock_generator.generate_summary.return_value = "Summary text"
        
        response = self.client.post("/summary", json={"repo_map": "Map content"})
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"], "Summary text")

    @patch('server.main.embedder')
    def test_embed_summary(self, mock_embedder):
        mock_embedder.embed_text.return_value = [0.1, 0.2]
        
        response = self.client.post("/embed/summary", json={"summary": "Summary text"})
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["embedding"], [0.1, 0.2])

if __name__ == '__main__':
    unittest.main()
