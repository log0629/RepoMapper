import sys
import os
import pytest
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import RepoSummaryGenerator, LLMClient
from repomap_class import RepoMap

class MockLLMClient:
    def complete(self, prompt: str) -> str:
        return "This is a mock summary of the repository."

@pytest.fixture
def repo_map_mock():
    # Mock RepoMap to avoid actual file system operations and graph building during this unit test
    repo_map = MagicMock(spec=RepoMap)
    repo_map.get_repo_map.return_value = ("src/main.py\nsrc/utils.py", None)
    return repo_map

def test_summary_generation(repo_map_mock):
    """Test that RepoSummaryGenerator correctly calls the LLM client."""
    generator = RepoSummaryGenerator(repo_map_mock)
    client = MockLLMClient()
    
    summary = generator.get_summary(client)
    
    assert summary == "This is a mock summary of the repository."
    
    # Verify get_repo_map was called
    repo_map_mock.get_repo_map.assert_called_once()

def test_prompt_construction():
    """Test that the prompt contains the repo map."""
    repo_map = MagicMock(spec=RepoMap)
    test_map = "file1.py\nfile2.py"
    repo_map.get_repo_map.return_value = (test_map, None)
    
    generator = RepoSummaryGenerator(repo_map)
    
    # Capture the prompt passed to client
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = "Summary"
    
    generator.get_summary(client)
    
    # Check arguments passed to complete
    args, _ = client.complete.call_args
    prompt = args[0]
    
    assert "Analyze the following repository map" in prompt
    assert test_map in prompt
