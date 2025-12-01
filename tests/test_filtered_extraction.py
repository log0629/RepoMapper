import sys
import os
import pytest
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repomap_class import RepoMap, ParsedTag

@pytest.fixture
def repo_map_with_tags():
    repo_map = RepoMap()
    
    # Mock methods to avoid file system and complex logic
    repo_map._calculate_file_ranks = MagicMock()
    repo_map.get_ranked_tags = MagicMock()
    repo_map.get_rel_fname = lambda f: f # Simple identity for test
    repo_map.to_tree = MagicMock()
    repo_map.token_count = MagicMock()
    
    # Setup dummy data
    # 5 tags, each taking 10 tokens (simulated)
    tags = []
    for i in range(5):
        tag = ParsedTag(
            rel_fname=f"file{i}.py",
            fname=f"/abs/file{i}.py",
            line=10*i,
            name=f"func{i}",
            kind="def",
            end_line=10*i+5,
            content=f"def func{i}():\n    pass"
        )
        # Rank is descending: 1.0, 0.9, ...
        tags.append((1.0 - i*0.1, tag))
    
    # Return ranks dict and list
    ranks_dict = {f"file{i}.py": 1.0 - i*0.1 for i in range(5)}
    repo_map._calculate_file_ranks.return_value = (ranks_dict, None, [f"file{i}.py" for i in range(5)])
    repo_map.get_ranked_tags.return_value = (tags, None)
    
    # Mock to_tree and token_count to simulate size
    # Assume each tag adds 10 tokens
    def mock_to_tree(selected_tags, chat_rel_fnames):
        return "\n".join([t[1].name for t in selected_tags])
    
    def mock_token_count(tree_str):
        # Simple count: 10 tokens per line (per tag)
        if not tree_str: return 0
        return len(tree_str.splitlines()) * 10
        
    repo_map.to_tree.side_effect = mock_to_tree
    repo_map.token_count.side_effect = mock_token_count
    
    return repo_map

def test_filtering_logic(repo_map_with_tags):
    """Test that get_semantic_blocks respects token_limit."""
    repo_map = repo_map_with_tags
    
    # Case 1: Limit allows all 5 tags (50 tokens)
    # limit = 60
    blocks = repo_map.get_semantic_blocks(other_fnames=["dummy"], token_limit=60)
    assert len(blocks) == 5
    assert blocks[0].name == "func0"
    assert blocks[4].name == "func4"
    
    # Case 2: Limit allows only 3 tags (30 tokens)
    blocks = repo_map.get_semantic_blocks(other_fnames=["dummy"], token_limit=35)
    assert len(blocks) == 3
    assert blocks[0].name == "func0"
    assert blocks[2].name == "func2"
    
    # Case 3: Limit allows only 1 tag
    blocks = repo_map.get_semantic_blocks(other_fnames=["dummy"], token_limit=15)
    assert len(blocks) == 1
    assert blocks[0].name == "func0"

def test_no_limit(repo_map_with_tags):
    """Test behavior without limit."""
    repo_map = repo_map_with_tags
    # Should return all (mocked get_tags needs to be handled if we go down that path, 
    # but here we mocked _calculate_file_ranks and get_tags is called inside the 'no limit' path)
    
    # Wait, the 'no limit' path calls self.get_tags(fname). 
    # We need to mock that too for the 'no limit' path to work in this unit test setup.
    repo_map.get_tags = MagicMock()
    
    # Setup get_tags to return the tag for the file
    def mock_get_tags(fname, rel_fname):
        # Extract index from fname "file{i}.py"
        idx = int(fname.split("file")[1].split(".")[0])
        return [ParsedTag(
            rel_fname=rel_fname,
            fname=fname,
            line=10*idx,
            name=f"func{idx}",
            kind="def",
            end_line=10*idx+5,
            content=f"def func{idx}():\n    pass"
        )]
    repo_map.get_tags.side_effect = mock_get_tags
    
    blocks = repo_map.get_semantic_blocks(other_fnames=["dummy"])
    assert len(blocks) == 5
