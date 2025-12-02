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

    def get_stored_block_ids(self, repo_id: str) -> set:
        """Fetch all existing block IDs for a repository."""
        stored_ids = set()
        offset = None
        while True:
            # Scroll through all points for this repo
            # Note: Qdrant scroll API might differ slightly based on version.
            # Using basic scroll with filter.
            points, next_offset = self.client.scroll(
                collection_name=COLLECTION_BLOCKS,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repo_id",
                            match=models.MatchValue(value=repo_id),
                        )
                    ]
                ),
                limit=100, # Batch size
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            
            for point in points:
                stored_ids.add(point.id)
                
            if next_offset is None:
                break
            offset = next_offset
            
        return stored_ids

    def index_repository_data(
        self, 
        repo_id: str, 
        commit_sha: str, 
        summary: str, 
        em_summary: List[float], 
        blocks: List[Dict[str, Any]]
    ):
        """Index repository data (summary and blocks) into Qdrant using Smart Diffing."""
        
        # 1. Upsert Repository Info (Always update summary/SHA)
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

        # 2. Smart Diffing for Blocks
        
        # A. Get Stored IDs
        stored_ids = self.get_stored_block_ids(repo_id)
        
        # B. Identify Current Blocks and Generate IDs
        current_blocks_map = {} # ID -> Block Data
        for block in blocks:
            # Generate ID based on content hash (deterministic)
            # Key includes: repo_id, file_path, name, start_line
            # Ideally should include content hash to detect content changes even if line numbers are same
            # But here we use path+name+line as identity. 
            # If content changes, we want to update.
            # Wait, if we use path+name+line as ID, then content change = SAME ID.
            # So upserting same ID with new vector = Update. Correct.
            
            # BUT: If we want to skip unchanged blocks, we need to know if content changed.
            # If ID is same, we assume it's the "same block entity".
            # To skip, we need to compare content hash?
            # Or: We can just upsert everything that exists in current.
            # The "Skip" benefit comes if we DON'T generate embeddings for unchanged blocks.
            # But here 'blocks' already have 'em_content' (embeddings).
            # So the embedding cost is already paid by the caller (server/main.py).
            
            # Wait, the user's concern was:
            # "Local cache saves parsing time. But if we delete & re-insert, we re-embed everything."
            # So server/main.py SHOULD NOT embed if block is unchanged.
            # But server/main.py doesn't know Qdrant state.
            
            # To fully realize the benefit, the "Diffing" should happen BEFORE embedding.
            # However, that requires 2 round trips to Indexer or Indexer logic inside Server.
            
            # Let's stick to the Indexer-side diffing for now to ensure DB consistency (Delete obsolete).
            # Optimization (Skip Embedding) is a further step.
            # For now, we implement:
            # - Delete: Stored - Current
            # - Upsert: Current (Updates existing, Adds new)
            
            # If we want to skip Upsert for unchanged blocks to save DB IO:
            # We can checks if ID exists in stored_ids.
            # But we don't know if content changed unless we check hash.
            # Since we are overwriting, it's safe.
            
            # Let's implement the Delete logic first, which is the critical correctness fix.
            
            block_id = self._generate_id(f"{repo_id}:{block['file_path']}:{block['name']}:{block['start_line']}")
            current_blocks_map[block_id] = block

        current_ids = set(current_blocks_map.keys())
        
        # C. Calculate Diff
        to_delete = stored_ids - current_ids
        to_upsert = current_ids # We upsert all current to ensure latest state (content/vector update)
        # Optimization: If we had content hash stored in payload, we could compare and skip upsert.
        
        print(f"Smart Diffing: Stored={len(stored_ids)}, Current={len(current_ids)}")
        print(f"Actions: Delete={len(to_delete)}, Upsert={len(to_upsert)}")

        # D. Delete Obsolete Blocks
        if to_delete:
            self.client.delete(
                collection_name=COLLECTION_BLOCKS,
                points_selector=models.PointIdsList(
                    points=list(to_delete)
                )
            )
            print(f"Deleted {len(to_delete)} obsolete blocks")

        # E. Upsert Current Blocks
        points = []
        for block_id, block in current_blocks_map.items():
            if 'em_content' not in block:
                print(f"ERROR: Block {block.get('name')} missing em_content")
                continue
                
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

    def search_repositories(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant repositories based on query vector."""
        try:
            results = self.client.search(
                collection_name=COLLECTION_REPOS,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            return [
                {
                    "repo_id": hit.payload.get("repo_id"),
                    "score": hit.score,
                    "summary": hit.payload.get("summary")
                }
                for hit in results
            ]
        except Exception as e:
            print(f"Error searching repositories: {e}")
            return []

    def search_code_blocks(self, query_vector: List[float], repo_ids: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant code blocks within specified repositories."""
        try:
            # Filter by repo_ids
            repo_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="repo_id",
                        match=models.MatchAny(any=repo_ids)
                    )
                ]
            ) if repo_ids else None

            results = self.client.search(
                collection_name=COLLECTION_BLOCKS,
                query_vector=query_vector,
                query_filter=repo_filter,
                limit=limit,
                with_payload=True
            )
            
            return [
                {
                    "repo_id": hit.payload.get("repo_id"),
                    "file_path": hit.payload.get("file_path"),
                    "name": hit.payload.get("name"),
                    "content": hit.payload.get("content"),
                    "start_line": hit.payload.get("start_line"),
                    "score": hit.score
                }
                for hit in results
            ]
        except Exception as e:
            print(f"Error searching code blocks: {e}")
            return []

    def _generate_id(self, key: str) -> str:
        """Generate a deterministic UUID from a string key."""
        hash_val = hashlib.md5(key.encode()).hexdigest()
        return str(uuid.UUID(hash_val))
