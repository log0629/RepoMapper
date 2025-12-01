import unittest
from unittest.mock import MagicMock, patch
from rag.llm_client import OpenAILLMClient

class TestOpenAILLMClient(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.llm_client = OpenAILLMClient(client=self.mock_client, model="gpt-4")

    def test_generate_text(self):
        # Mock response
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated text"
        self.mock_client.chat.completions.create.return_value.choices = [mock_choice]
        
        # Call method
        result = self.llm_client.generate_text("Test prompt")
        
        # Verify
        self.mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test prompt"}]
        )
        self.assertEqual(result, "Generated text")

if __name__ == '__main__':
    unittest.main()
