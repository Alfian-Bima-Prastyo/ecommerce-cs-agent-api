"""
Product Agent handle product inquiries, stock, specifications.
Collections: ecommerce_product_catalog
"""
from agents.base_agent import BaseAgent
from retrieval.hybrid_retriever import HybridRetriever

class ProductAgent(BaseAgent):
    agent_name = "product_agent"

    def __init__(self, retriever: HybridRetriever, **kwargs):
        super().__init__(retriever, **kwargs)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Kamu adalah agen produk e-commerce Indonesia yang membantu customer menemukan produk yang tepat.
Gunakan HANYA informasi dari konteks berikut untuk menjawab.

KONTEKS PRODUK:
{context}

PERTANYAAN CUSTOMER:
{query}

INSTRUKSI:
- Sebutkan nama produk, harga, dan spesifikasi utama jika relevan
- Format harga dalam rupiah (contoh: Rp150.000)
- Jika stok tidak disebutkan di konteks, jangan berasumsi
- Bahasa Indonesia sopan dan informatif
- Maksimal 3 paragraf

JAWABAN:"""