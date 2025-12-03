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
            {
                "ranking": "1",
                "project_name": "repo1",
                "project_url": "https://github.com/user/repo1",
                "stars": "100",
                "language": "Python"
            },
            {
                "ranking": "2",
                "project_name": "repo2",
                "project_url": "https://github.com/user/repo2",
                "stars": "50",
                "language": "Java"
            }
        ]

        # Call the endpoint
        response = client.post("/crawl/github-ranking")

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("data", data)
        self.assertEqual(len(data["data"]), 2)
        self.assertEqual(data["data"][0]["project_name"], "repo1")
        self.assertEqual(data["data"][0]["project_url"], "https://github.com/user/repo1")
        
        # Verify mock was called without limit
        mock_crawl.assert_called_once_with()

if __name__ == '__main__':
    unittest.main()
