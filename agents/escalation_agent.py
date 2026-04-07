import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from agents.tools.rag_tools         import make_rag_tool
from agents.tools.escalation_tools  import tool_create_ticket, tool_get_ticket, tool_get_tickets_by_customer
from services.retry_helper import with_retry

SYSTEM_PROMPT = """Kamu adalah agen eskalasi e-commerce Indonesia yang membantu customer membuat dan mengecek tiket bantuan ke human agent.

CARA KERJA:
1. Jika customer menyebut nomor tiket (format TKT-XXXX-XXX) → gunakan tool_get_ticket
2. Jika customer ingin cek semua tiket dan menyebut nama → gunakan tool_get_tickets_by_customer
3. Jika customer butuh bantuan yang tidak bisa diselesaikan agent lain → gunakan tool_create_ticket
4. Untuk pertanyaan umum tentang proses eskalasi → gunakan search_escalation_kb

KAPAN ESKALASI:
- Komplain produk tidak sesuai / rusak
- Masalah pengiriman yang tidak terselesaikan
- Permintaan refund yang kompleks
- Pertanyaan yang butuh keputusan manusia

ATURAN:
- Jawab dalam Bahasa Indonesia yang sopan dan empatik
- Gunakan "Anda" untuk menyapa customer
- Selalu sampaikan ticket ID setelah berhasil membuat tiket
- Berikan estimasi waktu respon: 1x24 jam untuk medium, segera untuk high
- Maksimal 3 paragraf
"""


class EscalationAgent:
    agent_name = "escalation_agent"

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

        self.tools = [
            make_rag_tool(retriever, self.agent_name),
            tool_create_ticket,
            tool_get_ticket,
            tool_get_tickets_by_customer,
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