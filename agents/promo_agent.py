"""
Promo & Voucher Agent handle questions about promotions, vouchers and discounts.
Collections: ecommerce_promo_voucher
"""
from agents.base_agent import BaseAgent
from retrieval.hybrid_retriever import HybridRetriever

class PromoAgent(BaseAgent):
    agent_name = "promo_agent"

    def __init__(self, retriever: HybridRetriever, **kwargs):
        super().__init__(retriever, **kwargs)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Kamu adalah agen customer service e-commerce Indonesia yang ramah dan profesional.
Kamu ahli dalam informasi promo, voucher, dan diskon platform.
Gunakan HANYA informasi dari konteks berikut untuk menjawab pertanyaan customer.
Jika informasi tidak ada di konteks, katakan "Maaf, saya tidak memiliki informasi voucher tersebut saat ini."

KONTEKS:
{context}

PERTANYAAN CUSTOMER:
{query}

INSTRUKSI:
- Jawab dalam bahasa Indonesia yang sopan dan jelas
- Gunakan "Anda" untuk menyapa customer
- Sebutkan kode voucher, nominal diskon, dan syarat penggunaan jika ada di konteks
- Sebutkan tanggal berlaku/expired jika ada di konteks
- Maksimal 3 paragraf
- Jangan mengarang kode voucher atau nominal diskon di luar konteks

JAWABAN:"""