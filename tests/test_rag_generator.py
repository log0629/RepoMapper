import unittest
from unittest.mock import MagicMock, patch
from rag.generator import RepoSummaryGenerator

class TestRepoSummaryGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MagicMock()
        self.generator = RepoSummaryGenerator(llm_client=self.mock_llm_client)

    def test_generate_summary(self):
        # Mock input
        repo_map_content = "File: main.py\nRank: 1.0\n..."
        
        # Mock LLM response
        self.mock_llm_client.generate_text.return_value = "This is a summary of the repository."
        
        # Call method
        summary = self.generator.generate_summary(repo_map_content)
        
        # Verify interactions
        self.mock_llm_client.generate_text.assert_called_once()
        self.assertEqual(summary, "This is a summary of the repository.")
        
    def test_generate_summary_empty_input(self):
        summary = self.generator.generate_summary("")
        self.assertEqual(summary, "")
        self.mock_llm_client.generate_text.assert_not_called()

if __name__ == '__main__':
    unittest.main()
