import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from services.retry_helper import with_retry
from agents.tools.rag_tools   import make_rag_tool
from agents.tools.promo_tools import tool_check_voucher_expiry, tool_validate_voucher, tool_get_active_promos, tool_apply_voucher

SYSTEM_PROMPT = """Kamu adalah agen promo e-commerce Indonesia yang membantu customer dengan voucher dan promo.

CARA KERJA:
1. Jika customer tanya voucher spesifik → gunakan tool_validate_voucher atau tool_check_voucher_expiry
2. Jika customer tanya promo aktif → gunakan tool_get_active_promos
3. Jika customer ingin tahu harga setelah voucher → gunakan tool_apply_voucher
4. Untuk info umum promo → gunakan search_promo_kb

ATURAN PENTING UNTUK tool_apply_voucher:
- cart_total HARUS berupa angka numerik, BUKAN ekspresi matematika
- Hitung dulu total harga sebelum memanggil tool
- Contoh BENAR: cart_total=190000
- Contoh SALAH: cart_total=2*95000

ATURAN UMUM:
- Jawab dalam Bahasa Indonesia yang sopan
- Gunakan "Anda" untuk menyapa customer
- Format harga dalam rupiah: Rp95.000
- Sebutkan syarat dan ketentuan voucher jika relevan
- Jika voucher tidak valid → jelaskan alasannya dengan jelas
- Maksimal 3 paragraf
"""


class PromoAgent:
    agent_name = "promo_agent"

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
        )

        self.tools = [
            make_rag_tool(retriever, self.agent_name),
            tool_validate_voucher,
            tool_check_voucher_expiry,
            tool_get_active_promos,
            tool_apply_voucher,
        ]

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SystemMessage(content=SYSTEM_PROMPT),
        )

    @with_retry(max_retries=2, backoff=2.0)
    def _invoke_with_retry(self, query: str, history: list = None) -> dict:
        messages = []
        for turn in (history or []):
            messages.append((turn["role"], turn["content"]))
        messages.append(("user", query))

        result     = self.agent.invoke({"messages": messages})
        msgs_out   = result["messages"]
        answer     = msgs_out[-1].content
        tools_used = [m.name for m in msgs_out if hasattr(m, "name") and m.name]

        return {
            "answer":     answer,
            "sources":    tools_used,
            "confidence": "high",
            "agent":      self.agent_name,
        }

    def run(self, query: str, history: list = None) -> dict:
        try:
            return self._invoke_with_retry(query, history)
        except Exception as e:
            print(f"[PROMO_AGENT] ERROR: {type(e).__name__}: {e}") 
            return {
                "answer":     "Maaf, terjadi kesalahan saat memproses permintaan Anda. Silakan coba lagi.",
                "sources":    [],
                "confidence": "high",
                "agent":      self.agent_name,
            }