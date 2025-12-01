from pydantic import BaseModel
from typing import List, Optional, Any

class RepoRequest(BaseModel):
    root_path: str
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

class SemanticBlockModel(BaseModel):
    file_path: str
    type: str
    name: str
    start_line: int
    end_line: int
    content: str
    rank_score: float
    em_content: Optional[List[float]] = None

class SemanticBlocksResponse(BaseModel):
    blocks: List[SemanticBlockModel]

class EmbedSummaryRequest(BaseModel):
    repo_map: str

class EmbedSummaryResponse(BaseModel):
    summary: str
    em_summary: List[float]

class EmbedBlocksRequest(BaseModel):
    blocks: List[SemanticBlockModel]

class EmbedBlocksResponse(BaseModel):
    blocks: List[SemanticBlockModel]
