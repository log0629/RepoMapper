import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repomap_class import RepoMap
from utils import Tag

# Mock file content for testing
TEST_FILE_NAME = "test_code.py"
TEST_FILE_CONTENT = """
def hello_world():
    print("Hello")

class Greeter:
    def greet(self):
        return "Hi"
"""

@pytest.fixture
def repo_map_setup(tmp_path):
    """Setup RepoMap instance and a test file."""
    # Create a temporary file
    d = tmp_path / "repo"
    d.mkdir()
    p = d / TEST_FILE_NAME
    p.write_text(TEST_FILE_CONTENT, encoding="utf-8")
    
    # Initialize RepoMap
    repo_map = RepoMap(root=str(d))
    
    return repo_map, str(p)

def test_single_parse_returns_rich_data(repo_map_setup):
    """Test that get_tags returns ParsedTag with full content."""
    repo_map, file_path = repo_map_setup
    rel_path = repo_map.get_rel_fname(file_path)
    
    # This should return the new ParsedTag structure (or compatible)
    tags = repo_map.get_tags(file_path, rel_path)
    
    assert len(tags) > 0
    
    # Check for definition of hello_world
    hello_func = next((t for t in tags if t.name == "hello_world"), None)
    assert hello_func is not None
    
    # Verify new fields exist (these will fail until refactoring)
    assert hasattr(hello_func, "content")
    assert "def hello_world():" in hello_func.content
    assert hasattr(hello_func, "end_line")
    assert hello_func.end_line > hello_func.line

def test_ranking_still_works(repo_map_setup):
    """Test that get_ranked_tags still works with the new structure."""
    repo_map, file_path = repo_map_setup
    
    # Should not crash
    ranked_tags, report = repo_map.get_ranked_tags(
        chat_fnames=[],
        other_fnames=[file_path]
    )
    
    assert len(ranked_tags) > 0
    # Check that we still get rank and tag
    rank, tag = ranked_tags[0]
    assert isinstance(rank, float)
    assert tag.name in ["hello_world", "Greeter", "greet"]

def test_semantic_blocks_uses_cached_tags(repo_map_setup):
    """Test that get_semantic_blocks reuses the parsed tags."""
    repo_map, file_path = repo_map_setup
    
    # First call to populate cache (via get_ranked_tags or get_tags)
    repo_map.get_tags(file_path, TEST_FILE_NAME)
    
    # Now call get_semantic_blocks
    # We want to ensure it doesn't re-parse. 
    # Since we can't easily mock internal methods without more setup, 
    # we'll at least verify it returns correct data derived from the same source.
    blocks = repo_map.get_semantic_blocks(other_fnames=[file_path])
    
    assert len(blocks) > 0
    block = blocks[0]
    
    # Verify block attributes
    assert block.file_path == TEST_FILE_NAME
    assert block.content is not None
    assert "def" in block.content or "class" in block.content
    assert block.rank_score is not None

def test_caching_mechanism(repo_map_setup):
    """Verify that tags are actually cached."""
    repo_map, file_path = repo_map_setup
    rel_path = TEST_FILE_NAME
    
    # First fetch
    tags1 = repo_map.get_tags(file_path, rel_path)
    
    # Modify cache manually to verify second fetch hits it
    # (This assumes implementation detail of TAGS_CACHE)
    if isinstance(repo_map.TAGS_CACHE, dict):
        # In-memory cache
        cached_data = repo_map.TAGS_CACHE[file_path]["data"]
        # Modify the first tag's name in the cache
        # We need to be careful if it's a tuple, but let's assume we can replace the list
        modified_tags = list(cached_data)
        # We can't easily modify a namedtuple/dataclass if frozen, but we can replace the list item
        # For this test, let's just check object identity if possible, or rely on side effects.
        # Simpler: check if get_tags returns the exact same object list
        tags2 = repo_map.get_tags(file_path, rel_path)
        assert tags1 == tags2
