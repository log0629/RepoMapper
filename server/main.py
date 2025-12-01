from fastapi import FastAPI, HTTPException
from .models import (
    RepoRequest, RepoMapResponse, 
    SemanticBlocksResponse, 
    EmbedSummaryRequest, EmbedSummaryResponse,
    EmbedBlocksRequest, EmbedBlocksResponse
)
from .manager import RepositoryManager
import os
from openai import OpenAI
from rag import RepoSummaryGenerator, Embedder, OpenAILLMClient

app = FastAPI(title="RepoMapper API")
manager = RepositoryManager()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. RAG features will fail.")
    OPENAI_API_KEY = "missing"

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Initialize RAG components
llm_client = OpenAILLMClient(client=client, model=LLM_MODEL)
generator = RepoSummaryGenerator(llm_client=llm_client)
# Embedder uses local model, no client needed
embedder = Embedder(model=EMBEDDING_MODEL or "all-MiniLM-L6-v2")
# Initialize Indexer
from rag.indexer import RepoIndexer
indexer = RepoIndexer()


@app.post("/repomap", response_model=RepoMapResponse)
async def get_repo_map(request: RepoRequest):
    try:
        content, commit_sha = manager.extract_repo_map(request)
        return RepoMapResponse(
            repo_map=content, 
            repo_id=request.repo_id,
            commit_sha=commit_sha
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/semantic-blocks", response_model=SemanticBlocksResponse)
async def get_semantic_blocks(request: RepoRequest):
    try:
        blocks, commit_sha = manager.extract_semantic_blocks(request)
        # Assign repo_id to blocks if provided in request
        if request.repo_id:
            for block in blocks:
                block['repo_id'] = request.repo_id
        return SemanticBlocksResponse(blocks=blocks, commit_sha=commit_sha)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/summary", response_model=EmbedSummaryResponse)
async def embed_summary(request: EmbedSummaryRequest):
    try:
        # 1. Generate Summary
        summary = generator.generate_summary(request.repo_map)
        
        # 2. Generate Embedding
        embedding = embedder.embed_text(summary)
        
        return EmbedSummaryResponse(
            summary=summary, 
            em_summary=embedding,
            repo_id=request.repo_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/blocks", response_model=EmbedBlocksResponse)
async def embed_blocks(request: EmbedBlocksRequest):
    try:
        # Extract contents for batch embedding
        contents = [block.content for block in request.blocks]
        
        # Generate embeddings
        embeddings = embedder.embed_batch(contents)
        
        # Assign embeddings back to blocks
        for block, embedding in zip(request.blocks, embeddings):
            block.em_content = embedding
            # If repo_id is provided in request, assign it to blocks if they don't have one
            if request.repo_id and not block.repo_id:
                block.repo_id = request.repo_id
            
        return EmbedBlocksResponse(blocks=request.blocks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_repository(request: RepoRequest):
    try:
        # 1. Check if repo needs indexing
        if not request.repo_id:
            raise HTTPException(status_code=400, detail="repo_id is required for indexing")
            
        last_sha = indexer.get_last_commit_sha(request.repo_id)
        
        # We need to get current SHA. We can get it by extracting the map (or just checking git)
        # Since we need the map anyway if we proceed, let's extract it.
        # But extracting map is expensive if we skip.
        # Ideally manager should have a lightweight 'get_current_sha' method.
        # For now, let's use extract_repo_map as it returns SHA.
        # Optimization: We could add a 'check_sha_only' flag to manager?
        # Or just use get_current_commit_sha directly here if we had access to the path.
        # But manager handles the path logic.
        
        # Let's assume we extract map first. If SHA matches, we discard it.
        # This is slightly inefficient but safe.
        # Better: Use manager to get SHA first.
        
        # For this implementation, we follow the plan:
        # 1. Get current SHA (via manager helper or full extraction)
        # Let's do full extraction for simplicity as per current manager API.
        
        content, current_sha = manager.extract_repo_map(request)
        
        if not current_sha:
             raise HTTPException(status_code=500, detail="Could not determine commit SHA")
             
        if last_sha == current_sha:
            return {"status": "skipped", "commit_sha": current_sha, "repo_id": request.repo_id}
            
        # 2. Proceed with Indexing
        
        # Generate Summary
        summary = generator.generate_summary(content)
        em_summary = embedder.embed_text(summary)
        
        # Extract Blocks
        blocks, _ = manager.extract_semantic_blocks(request)
        
        # Embed Blocks
        block_contents = [b['content'] for b in blocks]
        em_blocks = embedder.embed_batch(block_contents)
        
        # Assign embeddings to blocks
        for block, em in zip(blocks, em_blocks):
            block['em_content'] = em
            block['repo_id'] = request.repo_id # Ensure repo_id is set
            
        # Index Data
        indexer.index_repository_data(
            repo_id=request.repo_id,
            commit_sha=current_sha,
            summary=summary,
            em_summary=em_summary,
            blocks=blocks
        )
        
        return {"status": "indexed", "commit_sha": current_sha, "repo_id": request.repo_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start():
    import uvicorn
    # Create collections on startup
    try:
        indexer.create_collections()
    except Exception as e:
        print(f"Warning: Failed to create collections: {e}")
        
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
