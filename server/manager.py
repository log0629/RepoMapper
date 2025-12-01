from typing import Dict, List, Set
from core import RepoMap, find_src_files, count_tokens
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

    def extract_repo_map(self, request: RepoRequest) -> str:
        repo_map = self.get_repo_map_instance(request)
        
        # Resolve files
        # If other_files provided, use them. Else if not provided, find all files.
        # Note: repomap.py logic: if other_files or paths given, use them. Else find all.
        # Here we assume request.other_files contains the list if provided.
        
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
        return content or ""

    def extract_semantic_blocks(self, request: RepoRequest) -> List[dict]:
        # For semantic blocks, we also need a RepoMap instance. 
        # We can reuse the same logic but maybe we don't need all the chat_files etc for blocks?
        # Usually semantic blocks are extracted from specific files or all files.
        # The current implementation of get_semantic_blocks in RepoMap uses _calculate_file_ranks internally
        # which uses chat_files etc. So we should pass them if we want consistent ranking.
        
        repo_map = self.get_repo_map_instance(request)
        
        other_files = request.other_files
        if not other_files:
            other_files = find_src_files(request.root_path)
            
        blocks = repo_map.get_semantic_blocks(
            other_fnames=other_files, 
            token_limit=request.token_limit,
            # Note: get_semantic_blocks in repomap_class.py doesn't currently accept chat_files/mentioned args
            # It only takes other_fnames and token_limit. 
            # It calls _calculate_file_ranks(chat_fnames=[], other_fnames=other_fnames, ...) internally with defaults.
            # If we want to support chat_files influence on ranking for semantic blocks, 
            # we might need to update RepoMap.get_semantic_blocks signature.
            # For now, we stick to existing API of RepoMap.
        )
        return [asdict(b) for b in blocks]
