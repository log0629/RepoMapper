from fastapi import FastAPI, HTTPException
from .models import (
    RepoRequest, RepoMapResponse, 
    SemanticBlocksResponse, 
    SummaryRequest, SummaryResponse,
    EmbedSummaryRequest, EmbedSummaryResponse,
    EmbedBlocksRequest, EmbedBlocksResponse
)
from .manager import RepositoryManager
from rag import RepoSummaryGenerator, Embedder
from unittest.mock import MagicMock

app = FastAPI(title="RepoMapper API")
manager = RepositoryManager()

# Initialize RAG components (Mocking for now as per previous demo, 
# in production these would be initialized with real clients)
mock_llm = MagicMock()
mock_llm.generate_text.return_value = "This is a simulated summary."
generator = RepoSummaryGenerator(llm_client=mock_llm)

mock_embed_client = MagicMock()
mock_embed_client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1] * 1536)]
embedder = Embedder(client=mock_embed_client)


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

@app.post("/summary", response_model=SummaryResponse)
async def get_summary(request: SummaryRequest):
    try:
        summary = generator.generate_summary(request.repo_map)
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/summary", response_model=EmbedSummaryResponse)
async def embed_summary(request: EmbedSummaryRequest):
    try:
        embedding = embedder.embed_text(request.summary)
        return EmbedSummaryResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/blocks", response_model=EmbedBlocksResponse)
async def embed_blocks(request: EmbedBlocksRequest):
    try:
        embeddings = embedder.embed_batch(request.blocks)
        return EmbedBlocksResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start():
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
