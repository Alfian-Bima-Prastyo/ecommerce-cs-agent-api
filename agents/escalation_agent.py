"""
Escalation Agent handle complex cases that require escalation.
Collections: ecommerce_sop_escalation + ecommerce_ticket_history
"""
from agents.base_agent import BaseAgent
from retrieval.hybrid_retriever import HybridRetriever

class EscalationAgent(BaseAgent):
    agent_name = "escalation_agent"

    def __init__(self, retriever: HybridRetriever, **kwargs):
        super().__init__(retriever, model="qwen2.5:7b-instruct", **kwargs)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Kamu adalah agen eskalasi senior e-commerce Indonesia.
Tugasmu menangani kasus kompleks dan memastikan customer mendapat solusi terbaik.
Gunakan HANYA informasi dari konteks berikut.

KONTEKS (SOP eskalasi):
{context}

SITUASI CUSTOMER:
{query}

INSTRUKSI:
- Tunjukkan empati dan profesionalisme tinggi
- Jelaskan langkah eskalasi yang akan diambil
- Berikan estimasi waktu penyelesaian yang realistis
- Pastikan customer merasa didengar dan ditangani
- Bahasa Indonesia formal dan meyakinkan
- Maksimal 3 paragraf

JAWABAN:"""