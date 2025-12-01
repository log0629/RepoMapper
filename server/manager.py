from typing import Dict, List, Set, Tuple, Optional
from core import RepoMap, find_src_files, count_tokens, get_current_commit_sha
from dataclasses import asdict
from .models import RepoRequest

class RepositoryManager:
    def __init__(self):
        self.repos: Dict[str, RepoMap] = {}
        self.repo_models: Dict[str, str] = {} # Track model used for each repo

    def get_repo_map_instance(self, request: RepoRequest) -> RepoMap:
        root_path = request.root_path
        model = request.model
        
        # Check if we need to recreate the instance (e.g. if model changed)
        # For simplicity, if model changes, we recreate. 
        # If other params change, we can just update attributes.
        
        if root_path not in self.repos or self.repo_models.get(root_path) != model:
            def token_counter(text: str) -> int:
                return count_tokens(text, model)
                
            self.repos[root_path] = RepoMap(
                root=root_path, 
                map_tokens=request.token_limit,
                token_counter_func=token_counter,
                verbose=request.verbose,
                max_context_window=request.max_context_window,
                exclude_unranked=request.exclude_unranked
            )
            self.repo_models[root_path] = model
        else:
            # Update attributes of existing instance
            repo = self.repos[root_path]
            repo.map_tokens = request.token_limit
            repo.max_map_tokens = request.token_limit
            repo.verbose = request.verbose
            repo.max_context_window = request.max_context_window
            repo.exclude_unranked = request.exclude_unranked
            
        return self.repos[root_path]

    def extract_repo_map(self, request: RepoRequest) -> Tuple[str, Optional[str]]:
        repo_map = self.get_repo_map_instance(request)
        
        # Resolve files
        other_files = request.other_files
        if not other_files:
            other_files = find_src_files(request.root_path)
            
        # Convert sets
        mentioned_fnames = set(request.mentioned_files) if request.mentioned_files else None
        mentioned_idents = set(request.mentioned_idents) if request.mentioned_idents else None
        
        content, _ = repo_map.get_repo_map(
            chat_files=request.chat_files,
            other_files=other_files,
            mentioned_fnames=mentioned_fnames,
            mentioned_idents=mentioned_idents,
            force_refresh=request.force_refresh
        )
        
        # Get commit SHA
        commit_sha = get_current_commit_sha(request.root_path)
        
        return content or "", commit_sha

    def extract_semantic_blocks(self, request: RepoRequest) -> Tuple[List[dict], Optional[str]]:
        repo_map = self.get_repo_map_instance(request)
        
        other_files = request.other_files
        if not other_files:
            other_files = find_src_files(request.root_path)
            
        blocks = repo_map.get_semantic_blocks(
            other_fnames=other_files, 
            token_limit=request.token_limit,
        )
        
        # Get commit SHA
        commit_sha = get_current_commit_sha(request.root_path)
        
        return [asdict(b) for b in blocks], commit_sha
