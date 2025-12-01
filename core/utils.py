"""
Utility functions for RepoMap.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
from collections import namedtuple

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken is required. Install with: pip install tiktoken")
    sys.exit(1)

# Tag namedtuple for storing parsed code definitions and references
Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def read_text(filename: str, encoding: str = "utf-8", silent: bool = False) -> Optional[str]:
    """Read text from file with error handling."""
    try:
        return Path(filename).read_text(encoding=encoding, errors='ignore')
    except FileNotFoundError:
        if not silent:
            print(f"Error: {filename} not found.")
        return None
    except IsADirectoryError:
        if not silent:
            print(f"Error: {filename} is a directory.")
        return None
    except OSError as e:
        if not silent:
            print(f"Error reading {filename}: {e}")
        return None
    except UnicodeError as e:
        if not silent:
            print(f"Error decoding {filename}: {e}")
        return None
    except Exception as e:
        if not silent:
            print(f"An unexpected error occurred while reading {filename}: {e}")
        return None


def find_src_files(directory: str) -> List[str]:
    """Find source files in a directory."""
    if not os.path.isdir(directory):
        return [directory] if os.path.isfile(directory) else []
    
    src_files = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', 'env'}]
        
        for file in files:
            if not file.startswith('.'):
                full_path = os.path.join(root, file)
                src_files.append(full_path)
    
    return src_files


def get_current_commit_sha(repo_path: str) -> Optional[str]:
    """Get the current commit SHA of the repository."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repo or git not installed
        return None
