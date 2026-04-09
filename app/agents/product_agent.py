import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from agents.tools.rag_tools     import make_rag_tool
from agents.tools.product_tools import tool_check_stock, tool_get_product_price
from services.retry_helper import with_retry

SYSTEM_PROMPT = """Kamu adalah agen produk e-commerce Indonesia yang membantu customer menemukan produk yang tepat.

CARA KERJA:
1. Gunakan search_product_kb PERTAMA untuk cari informasi produk dari knowledge base
2. Dari hasil search_product_kb, ambil product_id (SKU) yang ada di payload
3. Jika customer menyebut nama produk (bukan SKU) → cari dulu via search_product_kb untuk dapat SKU, lalu gunakan SKU tersebut untuk cek stok atau harga
4. Jika customer sudah menyebut SKU spesifik → langsung gunakan tool_check_stock atau tool_get_product_price
5. Untuk perbandingan produk → cari kedua produk dengan search_product_kb lalu compare

ATURAN:
- Jawab dalam Bahasa Indonesia yang sopan dan informatif
- Gunakan "Anda" untuk menyapa customer
- Format harga dalam rupiah: Rp95.000
- Sebutkan nama produk, harga, spesifikasi utama jika relevan
- Jika stok tidak disebutkan di konteks → gunakan tool_check_stock dengan SKU yang didapat dari search
- Jangan berasumsi stok tersedia jika belum dicek
- PENTING: search_product_kb akan return payload yang berisi product_id — gunakan product_id itu untuk tool_check_stock
- Maksimal 3 paragraf
"""


class ProductAgent:
    agent_name = "product_agent"

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
            tool_check_stock,
            tool_get_product_price,
        ]

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SystemMessage(content=SYSTEM_PROMPT),
        )

    def run(self, query: str, history: list = None) -> dict:
        try:
            return self._invoke_with_retry(query, history)
        except Exception:
            return {
                "answer":     "Maaf, terjadi kesalahan saat memproses permintaan Anda. Silakan coba lagi.",
                "sources":    [],
                "confidence": "high",
                "agent":      self.agent_name,
            }

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