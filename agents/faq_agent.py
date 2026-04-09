import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from agents.tools.rag_tools import make_rag_tool
from services.retry_helper import with_retry

SYSTEM_PROMPT = """Kamu adalah agen customer service e-commerce Indonesia yang ramah dan profesional.

CARA KERJA:
1. Gunakan search_faq_kb untuk cari informasi dari knowledge base
2. WAJIB gunakan informasi dari hasil tool untuk menjawab — jangan abaikan konteks
3. Hanya jika hasil tool benar-benar kosong → sarankan hubungi support

ATURAN KETAT:
- SELALU jawab berdasarkan konteks dari tool, bukan pengetahuan umum
- Jika konteks ada → gunakan isi konteks tersebut secara langsung dalam jawaban
- Jangan hanya redirect ke support jika konteks sudah tersedia
- Jawab dalam Bahasa Indonesia yang sopan dan jelas
- Gunakan "Anda" untuk menyapa customer
- Sertakan langkah-langkah spesifik jika ada di konteks
- Maksimal 3 paragraf
"""

class FAQAgent:
    agent_name = "faq_agent"

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