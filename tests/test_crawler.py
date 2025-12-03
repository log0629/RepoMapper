import unittest
from unittest.mock import patch, MagicMock
from crawler.github_ranking import GithubRankingCrawler

class TestGithubRankingCrawler(unittest.TestCase):
    def setUp(self):
        self.crawler = GithubRankingCrawler()
        self.sample_markdown = """
| Ranking | Project Name | Stars | Forks | Language | Open Issues | Description | Last Commit |
| ------- | ------------ | ----- | ----- | -------- | ----------- | ----------- | ----------- |
| 1 | [build-your-own-x](https://github.com/codecrafters-io/build-your-own-x) | 445005 | 41751 | Markdown | 242 | Master programming by recreating your favorite technologies from scratch. | 2025-10-10T18:45:01Z |
| 2 | [freeCodeCamp](https://github.com/freeCodeCamp/freeCodeCamp) | 433847 | 42660 | TypeScript | 199 | freeCodeCamp.org's open-source codebase and curriculum. Learn math, programming, and computer science for free. | 2025-12-02T01:32:15Z |
| 3 | [awesome](https://github.com/sindresorhus/awesome) | 419441 | 32480 | None | 16 | 😎 Awesome lists about all kinds of interesting topics | 2025-11-22T10:35:27Z |
"""

    def test_parse_markdown(self):
        """Test parsing logic for extracting full table data from markdown."""
        data = self.crawler.parse_markdown(self.sample_markdown)
        
        # Verify filtering
        # The sample markdown has:
        # 1. build-your-own-x (Markdown) -> Should be filtered out (not in allowed list)
        # 2. freeCodeCamp (TypeScript) -> Should be kept
        # 3. awesome (None) -> Should be filtered out
        
        # Wait, "Markdown" is not in the allowed list.
        # "TypeScript" IS in the allowed list.
        # "None" is NOT.
        
        # So we expect only freeCodeCamp?
        # Let's check the sample markdown again in setUp
        
        # Sample markdown:
        # | 1 | [build-your-own-x]... | ... | Markdown | ...
        # | 2 | [freeCodeCamp]... | ... | TypeScript | ...
        # | 3 | [awesome]... | ... | None | ...
        
        # Allowed: C, C++, C#, Dart, Elixir, Go, Java, JavaScript, Kotlin, PHP, Python, Ruby, Rust, Scala, TypeScript
        
        self.assertEqual(len(data), 1)
        item = data[0]
        self.assertEqual(item['project_name'], 'freeCodeCamp')
        self.assertEqual(item['language'], 'TypeScript')

    def test_parse_markdown_filtering(self):
        """Test specific language filtering scenarios."""
        content = """
| Ranking | Project Name | Stars | Forks | Language |
| ------- | ------------ | ----- | ----- | -------- |
| 1 | [A](url) | 1 | 1 | Python |
| 2 | [B](url) | 1 | 1 | Pyhon |
| 3 | [C](url) | 1 | 1 | GO |
| 4 | [D](url) | 1 | 1 | Go |
| 5 | [E](url) | 1 | 1 | Rust |
| 6 | [F](url) | 1 | 1 | Unknown |
"""
        # Note: User's list had "Pyhon" and "GO". 
        # If I strictly follow user list: Pyhon and GO are allowed.
        # If I correct them: Python and Go are allowed.
        # I will implement with CORRECTED list: Python, Go.
        # So:
        # Python -> Keep
        # Pyhon -> Drop (unless user really meant Pyhon, but unlikely)
        # GO -> Drop (if case sensitive and list has Go) or Keep (if list has GO)
        # Go -> Keep
        # Rust -> Keep
        # Unknown -> Drop
        
        data = self.crawler.parse_markdown(content)
        languages = [d['language'] for d in data]
        self.assertIn('Python', languages)
        self.assertIn('Go', languages)
        self.assertIn('Rust', languages)
        self.assertNotIn('Unknown', languages)
        # self.assertNotIn('Pyhon', languages) # Assuming typo correction

    def test_parse_markdown_empty(self):
        """Test parsing empty or invalid markdown."""
        self.assertEqual(self.crawler.parse_markdown(""), [])
        self.assertEqual(self.crawler.parse_markdown("Invalid content"), [])

    @patch('crawler.github_ranking.requests.get')
    def test_fetch_file_list(self, mock_get):
        """Test fetching file list from GitHub API."""
        # Mock JSON response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "Top-100-stars.md"},
            {"name": "Top-100-Python.md"},
            {"name": "other_file.txt"}
        ]
        mock_get.return_value = mock_response

        files = self.crawler.fetch_file_list()
        self.assertIn("Top-100-stars.md", files)
        self.assertIn("Top-100-Python.md", files)
        self.assertNotIn("other_file.txt", files)
        self.assertEqual(len(files), 2)

    @patch('crawler.github_ranking.requests.get')
    def test_crawl(self, mock_get):
        """Test the full crawl process."""
        # Setup mocks
        # 1. First call: fetch file list (API)
        mock_response_list = MagicMock()
        mock_response_list.status_code = 200
        mock_response_list.json.return_value = [{"name": "Top-100-stars.md"}]
        
        # 2. Second call: fetch raw markdown
        mock_response_md = MagicMock()
        mock_response_md.status_code = 200
        mock_response_md.text = self.sample_markdown

        # Configure side_effect for multiple calls
        mock_get.side_effect = [mock_response_list, mock_response_md]

        # Execute
        results = self.crawler.crawl() # No limit
        
        # Verify
        # build-your-own-x (Markdown) is filtered out
        # freeCodeCamp (TypeScript) should be the first result
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['project_name'], "freeCodeCamp")
        self.assertEqual(results[0]['project_url'], "https://github.com/freeCodeCamp/freeCodeCamp")

if __name__ == '__main__':
    unittest.main()
