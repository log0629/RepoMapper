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
        """Test parsing logic for extracting GitHub URLs from markdown table."""
        urls = self.crawler.parse_markdown(self.sample_markdown)
        expected_urls = [
            "https://github.com/codecrafters-io/build-your-own-x",
            "https://github.com/freeCodeCamp/freeCodeCamp",
            "https://github.com/sindresorhus/awesome"
        ]
        self.assertEqual(urls, expected_urls)

    def test_parse_markdown_empty(self):
        """Test parsing empty or invalid markdown."""
        self.assertEqual(self.crawler.parse_markdown(""), [])
        self.assertEqual(self.crawler.parse_markdown("Invalid content"), [])

    @patch('crawler.github_ranking.requests.get')
    def test_fetch_file_list(self, mock_get):
        """Test fetching file list from GitHub page."""
        # Mock HTML response
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Simplified HTML structure mimicking GitHub file list
        mock_response.text = """
        <a href="/EvanLi/Github-Ranking/blob/master/Top100/Top-100-stars.md">Top-100-stars.md</a>
        <a href="/EvanLi/Github-Ranking/blob/master/Top100/Top-100-Python.md">Top-100-Python.md</a>
        """
        mock_get.return_value = mock_response

        files = self.crawler.fetch_file_list()
        self.assertIn("Top-100-stars.md", files)
        self.assertIn("Top-100-Python.md", files)

    @patch('crawler.github_ranking.requests.get')
    def test_crawl(self, mock_get):
        """Test the full crawl process."""
        # Setup mocks
        # 1. First call: fetch file list
        mock_response_list = MagicMock()
        mock_response_list.status_code = 200
        mock_response_list.text = '<a href="/EvanLi/Github-Ranking/blob/master/Top100/Top-100-stars.md">Top-100-stars.md</a>'
        
        # 2. Second call: fetch raw markdown
        mock_response_md = MagicMock()
        mock_response_md.status_code = 200
        mock_response_md.text = self.sample_markdown

        # Configure side_effect for multiple calls
        mock_get.side_effect = [mock_response_list, mock_response_md]

        # Execute
        results = self.crawler.crawl(limit=1) # Limit to 1 file for testing
        
        # Verify
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0], "https://github.com/codecrafters-io/build-your-own-x")

if __name__ == '__main__':
    unittest.main()
