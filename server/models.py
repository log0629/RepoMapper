from pydantic import BaseModel
from typing import List, Optional, Any

class RepoRequest(BaseModel):
    root_path: str
    repo_id: Optional[str] = None
    token_limit: int = 1024
    chat_files: List[str] = []
    other_files: List[str] = []
    mentioned_files: List[str] = []
    mentioned_idents: List[str] = []
    verbose: bool = False
    model: str = "gpt-4"
    max_context_window: Optional[int] = None
    force_refresh: bool = False
    exclude_unranked: bool = False

class RepoMapResponse(BaseModel):
    repo_map: str
    repo_id: Optional[str] = None
    commit_sha: Optional[str] = None

class SemanticBlockModel(BaseModel):
    file_path: str
    type: str
    name: str
    start_line: int
    end_line: int
    content: str
    rank_score: float
    em_content: Optional[List[float]] = None
    repo_id: Optional[str] = None

class SemanticBlocksResponse(BaseModel):
    blocks: List[SemanticBlockModel]
    commit_sha: Optional[str] = None

class EmbedSummaryRequest(BaseModel):
    repo_map: str
    repo_id: Optional[str] = None

class EmbedSummaryResponse(BaseModel):
    summary: str
    em_summary: List[float]
    repo_id: Optional[str] = None

class EmbedBlocksRequest(BaseModel):
    blocks: List[SemanticBlockModel]
    repo_id: Optional[str] = None

class EmbedBlocksResponse(BaseModel):
    blocks: List[SemanticBlockModel]
