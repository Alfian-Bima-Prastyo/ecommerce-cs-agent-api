"""
Order & Logistics Agent handle tracking, cancellation, shipping.
Collections: ecommerce_ticket_history + ecommerce_sop_escalation
"""
from agents.base_agent import BaseAgent
from retrieval.hybrid_retriever import HybridRetriever

class OrderAgent(BaseAgent):
    agent_name = "order_agent"

    def __init__(self, retriever: HybridRetriever, **kwargs):
        super().__init__(retriever, **kwargs)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Kamu adalah agen order & logistik e-commerce Indonesia.
Tugasmu membantu customer terkait pesanan, pengiriman, dan pembatalan.
Gunakan HANYA informasi dari konteks berikut.

KONTEKS (riwayat tiket & SOP):
{context}

PERTANYAAN CUSTOMER:
{query}

INSTRUKSI:
- Berikan langkah-langkah konkret yang bisa dilakukan customer
- Sebutkan estimasi waktu penyelesaian jika ada di konteks
- Jika kasus perlu eskalasi, informasikan dengan sopan
- Bahasa Indonesia formal dan menenangkan
- Maksimal 3 paragraf

JAWABAN:"""