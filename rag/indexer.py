import os
import hashlib
import uuid
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Collection Names
COLLECTION_REPOS = "repositories"
COLLECTION_BLOCKS = "code_blocks"

# Vector Size (all-MiniLM-L6-v2)
VECTOR_SIZE = 384

class RepoIndexer:
    def __init__(self, qdrant_url: str = QDRANT_URL, api_key: str = QDRANT_API_KEY):
        self.client = QdrantClient(url=qdrant_url, api_key=api_key)

    def create_collections(self):
        """Create collections if they don't exist."""
        # Repositories Collection
        if not self.client.collection_exists(COLLECTION_REPOS):
            self.client.create_collection(
                collection_name=COLLECTION_REPOS,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"Created collection: {COLLECTION_REPOS}")

        # Code Blocks Collection
        if not self.client.collection_exists(COLLECTION_BLOCKS):
            self.client.create_collection(
                collection_name=COLLECTION_BLOCKS,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"Created collection: {COLLECTION_BLOCKS}")

    def get_last_commit_sha(self, repo_id: str) -> Optional[str]:
        """Get the last processed commit SHA for a repository."""
        try:
            results = self.client.scroll(
                collection_name=COLLECTION_REPOS,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repo_id",
                            match=models.MatchValue(value=repo_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            if results[0]:
                return results[0][0].payload.get("commit_sha")
        except Exception as e:
            print(f"Error fetching commit SHA: {e}")
        return None

    def index_repository_data(
        self, 
        repo_id: str, 
        commit_sha: str, 
        summary: str, 
        em_summary: List[float], 
        blocks: List[Dict[str, Any]]
    ):
        """Index repository data (summary and blocks) into Qdrant."""
        
        # 1. Upsert Repository Info
        print(f"DEBUG: Upserting repo {repo_id}, vector size: {len(em_summary) if em_summary else 'None'}")
        self.client.upsert(
            collection_name=COLLECTION_REPOS,
            points=[
                models.PointStruct(
                    id=self._generate_id(repo_id),
                    vector=em_summary,
                    payload={
                        "repo_id": repo_id,
                        "summary": summary,
                        "commit_sha": commit_sha,
                    }
                )
            ]
        )
        print(f"Upserted repository summary for {repo_id}")

        # 2. Upsert Blocks
        points = []
        for block in blocks:
            # Generate a unique ID for the block
            block_id = self._generate_id(f"{repo_id}:{block['file_path']}:{block['name']}:{block['start_line']}")
            
            if 'em_content' not in block:
                print(f"ERROR: Block {block.get('name')} missing em_content")
                continue
            
            # print(f"DEBUG: Block {block.get('name')} vector size: {len(block['em_content'])}")
            
            points.append(models.PointStruct(
                id=block_id,
                vector=block["em_content"],
                payload={
                    "repo_id": repo_id,
                    "file_path": block["file_path"],
                    "name": block["name"],
                    "type": block["type"],
                    "content": block["content"],
                    "rank_score": block["rank_score"],
                    "start_line": block["start_line"],
                    "end_line": block["end_line"]
                }
            ))
        
        if points:
            self.client.upsert(
                collection_name=COLLECTION_BLOCKS,
                points=points
            )
            print(f"Upserted {len(points)} blocks for {repo_id}")

    def _generate_id(self, key: str) -> str:
        """Generate a deterministic UUID from a string key."""
        hash_val = hashlib.md5(key.encode()).hexdigest()
        return str(uuid.UUID(hash_val))
