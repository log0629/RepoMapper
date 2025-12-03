from fastapi.testclient import TestClient
from server.main import app
import unittest
from unittest.mock import patch

client = TestClient(app)

class TestServerCrawler(unittest.TestCase):
    @patch('crawler.github_ranking.GithubRankingCrawler.crawl')
    def test_crawl_endpoint(self, mock_crawl):
        # Setup mock return value
        mock_crawl.return_value = [
            "https://github.com/user/repo1",
            "https://github.com/user/repo2"
        ]

        # Call the endpoint
        response = client.post("/crawl/github-ranking?limit=5")

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("urls", data)
        self.assertEqual(len(data["urls"]), 2)
        self.assertEqual(data["urls"][0], "https://github.com/user/repo1")
        
        # Verify mock was called with correct limit
        mock_crawl.assert_called_once_with(limit=5)

if __name__ == '__main__':
    unittest.main()
