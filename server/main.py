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


@app.post("/repomap", response_model=RepoMapResponse)
async def get_repo_map(request: RepoRequest):
    try:
        content = manager.extract_repo_map(request)
        return RepoMapResponse(repo_map=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/semantic-blocks", response_model=SemanticBlocksResponse)
async def get_semantic_blocks(request: RepoRequest):
    try:
        blocks = manager.extract_semantic_blocks(request)
        return SemanticBlocksResponse(blocks=blocks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/summary", response_model=EmbedSummaryResponse)
async def embed_summary(request: EmbedSummaryRequest):
    try:
        # 1. Generate Summary
        summary = generator.generate_summary(request.repo_map)
        
        # 2. Generate Embedding
        embedding = embedder.embed_text(summary)
        
        return EmbedSummaryResponse(summary=summary, em_summary=embedding)
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
            
        return EmbedBlocksResponse(blocks=request.blocks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start():
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
