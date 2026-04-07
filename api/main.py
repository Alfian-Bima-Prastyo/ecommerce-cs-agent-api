import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from collections import defaultdict
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings
from retrieval.hybrid_retriever import HybridRetriever, AGENT_COLLECTIONS
from agents.orchestrator import Orchestrator

load_dotenv()

# Global state
retriever:    HybridRetriever = None
orchestrator: Orchestrator    = None

# Session memory
# Format: {session_id: [{"role": "user/assistant", "content": "..."}]}
session_store: dict = defaultdict(list)
MAX_HISTORY = 5

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
    query:      str
    session_id: str = ""  

class QueryResponse(BaseModel):
    query:      str
    intent:     str
    agent:      str
    answer:     str
    sources:    list[str]
    confidence: str
    escalate:   bool
    session_id: str 

class ProductItem(BaseModel):
    product_id:  str
    name:        str
    price:       int
    seller:      str
    category:    str
    rating:      float
    specs:       dict
    stock:       dict

class VoucherItem(BaseModel):
    code:         str
    description:  str
    discount:     dict
    min_purchase: int
    valid_until:  str
    quota:        int
    used:         int
    is_active:    bool

# Endpoints
# health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# chat endpoint
@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")
    session_id = request.session_id or str(uuid.uuid4())
    history = session_store[session_id]
    try:
        result = orchestrator.run(request.query, history)
        history.append({"role": "user",      "content": request.query})
        history.append({"role": "assistant", "content": result["answer"]})
        if len(history) > MAX_HISTORY * 2:
            session_store[session_id] = history[-(MAX_HISTORY * 2):]
        return QueryResponse(**result, session_id=session_id)
    except Exception as e:
        print(f"[API ERROR] {type(e).__name__}: {e}")  
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

# products endpoint
@app.get("/products", response_model=list[ProductItem])
def get_products():
    try:
        results, _ = retriever.dense.client.scroll(
            collection_name="ecommerce_product_catalog",
            limit=100,
            with_payload=True,
        )
        return [
            ProductItem(
                product_id=r.payload.get("product_id", ""),
                name=r.payload.get("name", ""),
                price=r.payload.get("price", 0),
                seller=r.payload.get("seller", ""),
                category=r.payload.get("category", ""),
                rating=r.payload.get("rating", 0.0),
                specs=r.payload.get("specs", {}),
                stock=r.payload.get("stock", {}),
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# vouchers endpoint
@app.get("/vouchers", response_model=list[VoucherItem])
def get_vouchers():
    try:
        results, _ = retriever.dense.client.scroll(
            collection_name="ecommerce_promo_voucher",
            scroll_filter=Filter(
                must=[FieldCondition(key="is_active", match=MatchValue(value=True))]
            ),
            limit=100,
            with_payload=True,
        )
        return [
            VoucherItem(
                code=r.payload.get("code", ""),
                description=r.payload.get("description", ""),
                discount=r.payload.get("discount", {}),
                min_purchase=r.payload.get("min_purchase", 0),
                valid_until=r.payload.get("valid_until", ""),
                quota=r.payload.get("quota", 0),
                used=r.payload.get("used", 0),
                is_active=r.payload.get("is_active", True),
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# product-page endpoint
@app.get("/products-page", response_class=HTMLResponse)
def products_page():
    results, _ = retriever.dense.client.scroll(
        collection_name="ecommerce_product_catalog",
        limit=100,
        with_payload=True,
    )
    
    cards = ""
    for r in results:
        p = r.payload
        cards += f"""
        <div class="card">
            <h3>{p.get('name')}</h3>
            <p class="sku">SKU: {p.get('product_id')}</p>
            <p class="price">Rp{p.get('price', 0):,}</p>
            <p>{p.get('description', '')[:100]}...</p>
            <p>⭐ {p.get('rating')} | {p.get('seller')}</p>
        </div>
        """
    
    return f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Katalog Produk</title>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ font-family: system-ui, sans-serif; background: #f5f5f5; padding: 24px; }}
            h1 {{ margin-bottom: 24px; color: #333; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
            .card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
            .card h3 {{ font-size: 15px; margin-bottom: 8px; color: #222; }}
            .sku {{ font-size: 12px; color: #888; margin-bottom: 4px; font-family: monospace; }}
            .price {{ font-size: 18px; font-weight: 600; color: #e53e3e; margin-bottom: 8px; }}
            p {{ font-size: 13px; color: #555; line-height: 1.5; margin-bottom: 4px; }}
        </style>
    </head>
    <body>
        <h1>📦 Katalog Produk</h1>
        <div class="grid">{cards}</div>
    </body>
    </html>
    """

# vouchers-page endpoint
@app.get("/vouchers-page", response_class=HTMLResponse)
def vouchers_page():
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    results, _ = retriever.dense.client.scroll(
        collection_name="ecommerce_promo_voucher",
        scroll_filter=Filter(
            must=[FieldCondition(key="is_active", match=MatchValue(value=True))]
        ),
        limit=100,
        with_payload=True,
    )
    
    cards = ""
    for r in results:
        v = r.payload
        discount = v.get("discount", {})
        if discount.get("type") == "persen":
            diskon_str = f"Diskon {discount['value']}%"
        elif discount.get("type") == "nominal":
            diskon_str = f"Potongan Rp{discount['value']:,}"
        else:
            diskon_str = "Gratis ongkir"
        
        sisa = v.get("quota", 0) - v.get("used", 0)
        cards += f"""
        <div class="card">
            <div class="badge">{diskon_str}</div>
            <h3>{v.get('code')}</h3>
            <p>{v.get('description')}</p>
            <p>Min. belanja: Rp{v.get('min_purchase', 0):,}</p>
            <p>Berlaku hingga: {v.get('valid_until')}</p>
            <p class="quota">Kuota tersisa: {sisa}</p>
        </div>
        """
    
    return f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Voucher Aktif</title>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ font-family: system-ui, sans-serif; background: #f5f5f5; padding: 24px; }}
            h1 {{ margin-bottom: 24px; color: #333; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
            .card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
            .badge {{ display: inline-block; background: #e53e3e; color: white; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-bottom: 10px; }}
            .card h3 {{ font-size: 20px; font-weight: 700; color: #222; margin-bottom: 8px; font-family: monospace; }}
            p {{ font-size: 13px; color: #555; line-height: 1.5; margin-bottom: 4px; }}
            .quota {{ color: #e53e3e; font-weight: 500; }}
        </style>
    </head>
    <body>
        <h1>🎟️ Voucher Aktif</h1>
        <div class="grid">{cards}</div>
    </body>
    </html>
    """