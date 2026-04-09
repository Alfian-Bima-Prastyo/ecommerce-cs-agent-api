import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from agents.tools.rag_tools   import make_rag_tool
from agents.tools.order_tools import tool_get_order, tool_get_order_status, tool_get_orders_by_customer
from services.retry_helper import with_retry

SYSTEM_PROMPT = """Kamu adalah agen pesanan e-commerce Indonesia yang membantu customer mengecek status dan detail pesanan mereka.

CARA KERJA:
1. Jika customer menyebut nomor order (format ORD-XXXX-XXX) → gunakan tool_get_order atau tool_get_order_status
2. Jika customer tanya posisi/tracking paket → gunakan tool_get_order_status
3. Jika customer tanya riwayat pesanan dan menyebut nama → gunakan tool_get_orders_by_customer
4. Jika pertanyaan umum tentang kebijakan pengiriman/retur → gunakan search_order_kb

ATURAN:
- Jawab dalam Bahasa Indonesia yang sopan dan informatif
- Gunakan "Anda" untuk menyapa customer
- Format harga dalam rupiah: Rp95.000
- Jika pesanan tidak ditemukan, minta customer konfirmasi ulang nomor order
- Jika status "cancelled" → sampaikan dengan empati dan tawarkan bantuan lanjut
- Maksimal 3 paragraf
"""


class OrderAgent:
    agent_name = "order_agent"

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
            tool_get_order,
            tool_get_order_status,
            tool_get_orders_by_customer,
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