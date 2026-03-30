"""
FastAPI layer for Agentic RAG E-Commerce Customer Service.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from retrieval.hybrid_retriever import HybridRetriever, AGENT_COLLECTIONS
from agents.orchestrator import Orchestrator

load_dotenv()

# Global state
retriever:    HybridRetriever = None
orchestrator: Orchestrator    = None

ALL_COLLECTIONS = AGENT_COLLECTIONS["all"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, orchestrator

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        model_kwargs={"device": "cpu"},
    )

    print("Connecting to Qdrant...")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )

    print("Building BM25 indexes...")
    retriever = HybridRetriever(client, embeddings)
    retriever.sparse.preload_all(ALL_COLLECTIONS)  
    print("Initializing orchestrator...")
    orchestrator = Orchestrator(retriever)

    print("API ready.")
    yield

# App
app = FastAPI(
    title="E-Commerce Customer Service API",
    version="1.0.0",
    lifespan=lifespan,
)

# Schema
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query:      str
    intent:     str
    agent:      str
    answer:     str
    sources:    list[str]
    confidence: str
    escalate:   bool

# Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")
    try:
        result = orchestrator.run(request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))