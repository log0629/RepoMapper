import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys

# Add project root to path to import scheduler if it were a module, 
# but since it's a script, we might import it dynamically or structure it as a module.
# For now, let's assume we will create a 'scheduler' module or import from the script.
# To make it testable, I'll assume we'll create a `scheduler.py` that can be imported.

# We will mock the functions in scheduler.py
# Since scheduler.py doesn't exist yet, we can't import it directly in the test file 
# without it existing. But I can define the test assuming the structure.

class TestScheduler(unittest.TestCase):
    def setUp(self):
        # Mock schedule module since it might not be installed
        self.mock_schedule = MagicMock()
        sys.modules['schedule'] = self.mock_schedule

    @patch('scheduler.requests.post')
    @patch('scheduler.subprocess.run')
    @patch('scheduler.os.path.exists')
    @patch('scheduler.os.makedirs')
    def test_run_job(self, mock_makedirs, mock_exists, mock_subprocess, mock_post):
        # Import the module inside test to avoid import error before creation
        # But since I am writing the test FIRST, I need to create the file first or 
        # use a placeholder. I will create the file in the next step.
        # Here I define what I EXPECT the module to do.
        
        # Ensure scheduler is re-imported if already imported
        if 'scheduler' in sys.modules:
            del sys.modules['scheduler']
            
        import scheduler
        from scheduler import run_job
        
        # Setup mocks
        # 1. Crawl response
        mock_crawl_response = MagicMock()
        mock_crawl_response.status_code = 200
        mock_crawl_response.json.return_value = {
            "data": [
                {
                    "project_name": "repo1",
                    "project_url": "https://github.com/org1/repo1",
                    "language": "Python"
                },
                {
                    "project_name": "repo2",
                    "project_url": "https://github.com/org2/repo2",
                    "language": "Java"
                }
            ]
        }
        
        # 2. Index response
        mock_index_response = MagicMock()
        mock_index_response.status_code = 200
        
        mock_post.side_effect = [mock_crawl_response, mock_index_response, mock_index_response]
        
        # 3. File system
        # Assume repo1 exists (skip clone), repo2 does not (clone)
        mock_exists.side_effect = [True, False] 
        
        # Run the job
        run_job()
        
        # Verify Crawl
        mock_post.assert_any_call("http://localhost:8000/crawl/github-ranking")
        
        # Verify Clone for repo2 only
        # repo1 path check
        # repo2 path check
        
        # Verify Git operations
        # repo1 exists -> git pull
        expected_repo1_path = os.path.abspath("data/repos/org1_repo1")
        mock_subprocess.assert_any_call(
            ["git", "-C", expected_repo1_path, "pull"],
            check=True
        )

        # repo2 does not exist -> git clone
        expected_repo2_path = os.path.abspath("data/repos/org2_repo2")
        mock_subprocess.assert_any_call(
            ["git", "clone", "https://github.com/org2/repo2", expected_repo2_path],
            check=True
        )
        
        # Verify Index calls
        # We expect 2 index calls
        expected_repo1_path = os.path.abspath("data/repos/org1_repo1")
        
        # Check calls to index
        # We can't easily check exact order mixed with crawl, but we can check call args
        
        # Index call for repo1
        mock_post.assert_any_call(
            "http://localhost:8000/index",
            json={
                "root_path": expected_repo1_path
            }
        )
        
        # Index call for repo2
        mock_post.assert_any_call(
            "http://localhost:8000/index",
            json={
                "root_path": expected_repo2_path
            }
        )

if __name__ == '__main__':
    unittest.main()
