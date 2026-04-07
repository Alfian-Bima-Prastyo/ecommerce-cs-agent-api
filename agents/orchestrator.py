import os
import json
import time
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from agents.faq_agent        import FAQAgent
from agents.product_agent    import ProductAgent
from agents.order_agent      import OrderAgent
from agents.escalation_agent import EscalationAgent
from agents.promo_agent      import PromoAgent
from retrieval.hybrid_retriever import HybridRetriever
from langsmith import traceable


# State
class AgentState(TypedDict):
    query:      str
    intent:     str
    agent:      str
    context:    list
    answer:     str
    sources:    list
    confidence: str
    escalate:   bool
    history:    list
    is_multi:   bool        
    sub_tasks:  list        


INTENT_TO_AGENT = {
    # Small talk
    "small_talk":            "small_talk",
    "greeting":              "small_talk",
    "goodbye":               "small_talk",
    "thanks":                "small_talk",
    "out_of_scope":          "small_talk",

    # FAQ agent
    "cancel_order":          "faq_agent",
    "check_refund_policy":   "faq_agent",
    "delivery_period":       "faq_agent",
    "flash_sale_info":       "faq_agent",
    "check_payment_options": "faq_agent",
    "create_account":        "faq_agent",
    "delete_account":        "faq_agent",
    "edit_account":          "faq_agent",
    "get_invoice":           "faq_agent",
    "check_invoices":        "faq_agent",

    # Promo agent
    "check_voucher":         "promo_agent",
    "check_promo":           "promo_agent",
    "promo_info":            "promo_agent",
    "voucher_info":          "promo_agent",

    # Order agent
    "track_order":           "order_agent",
    "change_order":          "order_agent",
    "delivery_options":      "order_agent",
    "cod_payment":           "order_agent",
    "order_status":          "order_agent",

    # Product agent
    "check_stock":           "product_agent",
    "product_info":          "product_agent",
    "search_product":        "product_agent",
    "compare_products":      "product_agent",
    "check_price":           "product_agent",

    # Escalation agent
    "get_refund":            "escalation_agent",
    "return_request":        "escalation_agent",
    "seller_complaint":      "escalation_agent",
    "payment_issue":         "escalation_agent",
    "complaint":             "escalation_agent",
    "escalate_to_human":     "escalation_agent",
}

INTENT_LIST = ", ".join(INTENT_TO_AGENT.keys())

SMALL_TALK_RESPONSES = {
    "greeting":     "Halo! Selamat datang. Ada yang bisa saya bantu hari ini? 😊",
    "goodbye":      "Terima kasih telah menghubungi kami. Semoga hari Anda menyenangkan! 👋",
    "thanks":       "Sama-sama! Senang bisa membantu Anda. Ada hal lain yang ingin ditanyakan?",
    "small_talk":   "Halo! Saya asisten CS e-commerce. Saya siap membantu pertanyaan seputar pesanan, produk, voucher, dan pengiriman. Ada yang bisa saya bantu?",
    "out_of_scope": "Maaf, saya hanya dapat membantu pertanyaan seputar layanan e-commerce kami seperti pesanan, produk, voucher, dan pengiriman. Ada yang bisa saya bantu?",
}


class Orchestrator:
    def __init__(self, retriever: HybridRetriever):
        self.agents = {
            "faq_agent":        FAQAgent(retriever),
            "product_agent":    ProductAgent(retriever),
            "order_agent":      OrderAgent(retriever),
            "escalation_agent": EscalationAgent(retriever),
            "promo_agent":      PromoAgent(retriever),
        }

        self.router_llm = ChatGroq(
            model=os.getenv("GROQ_MODEL_PROD", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

        self.graph = self._build_graph()

    def _plan(self, state: AgentState) -> AgentState:
        prompt = f"""Analisis pertanyaan customer berikut.
    Tentukan apakah pertanyaan ini membutuhkan lebih dari satu domain untuk dijawab.

    Domain yang tersedia:
    - product: stok, harga, spesifikasi produk
    - promo: voucher, diskon, promo
    - order: status pesanan, tracking
    - faq: kebijakan, prosedur umum
    - escalation: komplain, retur, refund

    Pertanyaan: {state['query']}

    PENTING — is_multi=true HANYA jika pertanyaan eksplisit menyebut 2 domain berbeda sekaligus.
    Contoh multi: "cek stok SKU-10001 DAN cek voucher WELCOME10"
    Contoh single: "pakai voucher WELCOME10 untuk belanja Rp500.000" → ini single task domain promo
    Contoh single: "berapa harga SKU-10001 setelah voucher?" → ini single task domain promo

    Jawab dalam format JSON:
    {{
    "is_multi": true/false,
    "sub_tasks": [
        {{"task": "deskripsi task 1", "domain": "product/promo/order/faq/escalation"}},
        {{"task": "deskripsi task 2", "domain": "product/promo/order/faq/escalation"}}
    ]
    }}

    Jika single task, is_multi=false dan sub_tasks berisi 1 item.
    Jawab HANYA dengan JSON, tidak ada teks lain."""
        ...

        try:
            response = self.router_llm.invoke(prompt)
            content  = response.content.strip()
            content  = content.replace("```json", "").replace("```", "").strip()
            parsed   = json.loads(content)
            is_multi = parsed.get("is_multi", False)
            sub_tasks = parsed.get("sub_tasks", [])
        except Exception:
            is_multi  = False
            sub_tasks = []

        return {**state, "is_multi": is_multi, "sub_tasks": sub_tasks}

    def _route_after_plan(self, state: AgentState) -> str:
        """Route ke multi_task handler atau classify_intent biasa."""
        if state.get("is_multi") and len(state.get("sub_tasks", [])) > 1:
            return "multi_task"
        return "classify_intent"

    def _run_multi_task(self, state: AgentState) -> AgentState:
        sub_tasks    = state.get("sub_tasks", [])
        all_answers  = []
        all_sources  = []
        step_context = []  

        domain_to_agent = {
            "product":    "product_agent",
            "promo":      "promo_agent",
            "order":      "order_agent",
            "faq":        "faq_agent",
            "escalation": "escalation_agent",
        }

        for i, task in enumerate(sub_tasks):
            if i > 0:
                print(f"[MULTI_TASK] sleeping 5s before task {i+1}")
                time.sleep(5)

            domain     = task.get("domain", "")
            task_query = task.get("task", state["query"])
            agent_key  = domain_to_agent.get(domain)

            print(f"[MULTI_TASK] task {i+1}: domain={domain} agent={agent_key} query={task_query}")
            print(f"[MULTI_TASK] step_context so far: {step_context}")

            if not agent_key or agent_key not in self.agents:
                print(f"[MULTI_TASK] skip — agent not found")
                continue

            if step_context:
                context_str    = "\n".join(step_context)
                enriched_query = f"{task_query}\n\nKonteks dari langkah sebelumnya:\n{context_str}"
            else:
                enriched_query = task_query

            print(f"[MULTI_TASK] enriched_query: {enriched_query}")

            try:
                result = self.agents[agent_key].run(enriched_query, state["history"])
                print(f"[MULTI_TASK] result: {result['answer'][:150]}")
            except Exception as e:
                print(f"[MULTI_TASK] ERROR: {e}")
                result = {"answer": "Maaf, terjadi kesalahan.", "sources": []}

            all_answers.append(f"**{task_query}**\n{result['answer']}")
            all_sources.extend(result.get("sources", []))
            step_context.append(f"- {task_query}: {result['answer'][:200]}")

        combined_answer = "\n\n".join(all_answers) if all_answers else "Maaf, tidak dapat memproses permintaan."

        return {
            **state,
            "intent":     "multi_task",
            "agent":      "multi_agent",
            "answer":     combined_answer,
            "sources":    list(set(all_sources)),
            "confidence": "high",
            "escalate":   False,
        }

    def _classify_intent(self, state: AgentState) -> AgentState:
        prompt = f"""Klasifikasikan intent dari pertanyaan customer berikut.
Pilih SATU intent dari daftar ini:
{INTENT_LIST}

Panduan klasifikasi small talk:
- greeting: salam pembuka (halo, hai, selamat pagi, hi)
- goodbye: salam penutup (sampai jumpa, bye, dadah, selamat tinggal)
- thanks: ungkapan terima kasih (makasih, terima kasih, thanks)
- out_of_scope: pertanyaan di luar e-commerce (cuaca, masak, dll)
- small_talk: percakapan umum lainnya

Panduan klasifikasi voucher/promo:
- check_voucher: customer menyebut kode voucher spesifik
- voucher_info: customer tanya cara pakai voucher atau harga setelah voucher
- check_promo: customer tanya promo yang sedang berlaku

Pertanyaan: {state['query']}

Jawab HANYA dengan nama intent, tidak ada teks lain."""

        response = self.router_llm.invoke(prompt)
        intent   = response.content.strip().lower().replace(" ", "_")

        if intent not in INTENT_TO_AGENT:
            intent = "small_talk"

        agent = INTENT_TO_AGENT[intent]
        return {**state, "intent": intent, "agent": agent}

    def _run_small_talk(self, state: AgentState) -> AgentState:
        intent = state["intent"]
        answer = SMALL_TALK_RESPONSES.get(intent, SMALL_TALK_RESPONSES["small_talk"])
        return {
            **state,
            "answer":     answer,
            "sources":    [],
            "confidence": "high",
            "escalate":   False,
        }

    def _route_to_agent(self, state: AgentState) -> str:
        return state["agent"]

    def _run_faq_agent(self, state: AgentState) -> AgentState:
        result = self.agents["faq_agent"].run(state["query"], state["history"])
        return {**state, **result}

    def _run_product_agent(self, state: AgentState) -> AgentState:
        result = self.agents["product_agent"].run(state["query"], state["history"])
        return {**state, **result}

    def _run_order_agent(self, state: AgentState) -> AgentState:
        result = self.agents["order_agent"].run(state["query"], state["history"])
        return {**state, **result}

    def _run_escalation_agent(self, state: AgentState) -> AgentState:
        result = self.agents["escalation_agent"].run(state["query"], state["history"])
        escalate = result.get("confidence") == "low"
        return {**state, **result, "escalate": escalate}

    def _run_promo_agent(self, state: AgentState) -> AgentState:
        result = self.agents["promo_agent"].run(state["query"], state["history"])
        return {**state, **result}
    
    def _reflect(self, state: AgentState) -> AgentState:
        """Evaluasi apakah jawaban sudah menjawab pertanyaan user."""
        print(f"[REFLECT] agent={state['agent']} answer_len={len(state['answer'])} answer={state['answer'][:80]}")
        
        if state["agent"] in ["small_talk", "multi_agent"]:
            print(f"[REFLECT] skip — small_talk/multi_agent")
            return state

        if state["answer"] == "Maaf, terjadi kesalahan saat memproses permintaan Anda. Silakan coba lagi.":
            print(f"[REFLECT] error message detected → retry")
            time.sleep(5) 
            agent_key = state["agent"]
            if agent_key not in self.agents:
                return state
            try:
                result = self.agents[agent_key].run(state["query"], state["history"])
                print(f"[REFLECT] retry result: {result['answer'][:80]}")
                return {**state, **result}
            except Exception as e:
                print(f"[REFLECT] retry failed: {e}")
                return state

        prompt = f"""Evaluasi apakah jawaban berikut sudah menjawab pertanyaan customer dengan baik.

    Pertanyaan customer: {state['query']}
    Jawaban agent: {state['answer']}

    Jawab dalam format JSON:
    {{
    "is_good": true/false,
    "reason": "alasan singkat",
    "suggestion": "saran perbaikan jika is_good=false, kosong jika is_good=true"
    }}

    Kriteria jawaban yang baik:
    - Menjawab pertanyaan secara langsung
    - Menggunakan data spesifik (harga, stok, dll) bukan jawaban generik
    - Tidak hanya redirect ke support tanpa informasi
    - Dalam Bahasa Indonesia

    Jawab HANYA dengan JSON."""

        try:
            response = self.router_llm.invoke(prompt)
            content  = response.content.strip().replace("```json", "").replace("```", "").strip()
            parsed   = json.loads(content)
            is_good  = parsed.get("is_good", True)
            suggestion = parsed.get("suggestion", "")
        except Exception:
            return state

        if is_good:
            return state

        print(f"[REFLECT] Answer not good, retrying with suggestion: {suggestion}")
        agent_key = state["agent"]
        if agent_key not in self.agents:
            return state

        enriched_query = f"{state['query']}\n\nCatatan: {suggestion}"
        try:
            result = self.agents[agent_key].run(enriched_query, state["history"])
            return {**state, **result}
        except Exception:
            return state

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("plan",             self._plan)
        graph.add_node("classify_intent",  self._classify_intent)
        graph.add_node("small_talk",       self._run_small_talk)
        graph.add_node("multi_task",       self._run_multi_task)
        graph.add_node("faq_agent",        self._run_faq_agent)
        graph.add_node("product_agent",    self._run_product_agent)
        graph.add_node("order_agent",      self._run_order_agent)
        graph.add_node("escalation_agent", self._run_escalation_agent)
        graph.add_node("promo_agent",      self._run_promo_agent)
        graph.add_node("reflect",          self._reflect)

        graph.set_entry_point("plan")

        graph.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "multi_task":      "multi_task",
                "classify_intent": "classify_intent",
            }
        )

        graph.add_conditional_edges(
            "classify_intent",
            self._route_to_agent,
            {
                "faq_agent":        "faq_agent",
                "product_agent":    "product_agent",
                "order_agent":      "order_agent",
                "escalation_agent": "escalation_agent",
                "promo_agent":      "promo_agent",
                "small_talk":       "small_talk",
            }
        )

        graph.add_edge("faq_agent",        "reflect")
        graph.add_edge("product_agent",    "reflect")
        graph.add_edge("order_agent",      "reflect")
        graph.add_edge("escalation_agent", "reflect")
        graph.add_edge("promo_agent",      "reflect")
        graph.add_edge("small_talk",       "reflect")

        graph.add_edge("reflect",    END)
        graph.add_edge("multi_task", END)

        return graph.compile()

    @traceable(name="ecommerce-rag-orchestrator", run_type="chain")
    def run(self, query: str, history: list = None) -> dict:
        initial_state = AgentState(
            query=query,
            intent="",
            agent="",
            context=[],
            answer="",
            sources=[],
            confidence="",
            escalate=False,
            history=history or [],
            is_multi=False,
            sub_tasks=[],
        )
        result = self.graph.invoke(initial_state)
        return {
            "query":      result["query"],
            "intent":     result["intent"],
            "agent":      result["agent"],
            "answer":     result["answer"],
            "sources":    result["sources"],
            "confidence": result["confidence"],
            "escalate":   result["escalate"],
        }