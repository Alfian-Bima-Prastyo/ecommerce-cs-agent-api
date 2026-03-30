"""
FAQ & Policy Agent handle frequently asked questions and platform policies.
Collections: ecommerce_faq_policy
"""
from agents.base_agent import BaseAgent
from retrieval.hybrid_retriever import HybridRetriever

class FAQAgent(BaseAgent):
    agent_name = "faq_agent"

    def __init__(self, retriever: HybridRetriever, **kwargs):
        super().__init__(retriever, **kwargs)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Kamu adalah agen customer service e-commerce Indonesia yang ramah dan profesional.
Gunakan HANYA informasi dari konteks berikut untuk menjawab pertanyaan customer.
Jika informasi tidak ada di konteks, katakan "Maaf, saya tidak memiliki informasi tersebut."

KONTEKS:
{context}

PERTANYAAN CUSTOMER:
{query}

INSTRUKSI:
- Jawab dalam bahasa Indonesia yang sopan dan jelas
- Gunakan "Anda" untuk menyapa customer
- Sertakan angka/estimasi spesifik jika ada di konteks
- Maksimal 3 paragraf
- Jangan mengarang informasi di luar konteks

JAWABAN:"""