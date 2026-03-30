"""
LangGraph Orchestrator: router + planner + memory.
Determines which agent handles queries based on intent.
"""
import os
from typing import TypedDict, Annotated
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

# Intent classifier 
INTENT_TO_AGENT = {
    # FAQ agent
    "cancel_order":         "faq_agent",
    "check_refund_policy":  "faq_agent",
    "delivery_period":      "faq_agent",
    "flash_sale_info":      "faq_agent",
    "check_payment_options":"faq_agent",
    "create_account":       "faq_agent",
    "delete_account":       "faq_agent",
    "edit_account":         "faq_agent",
    "get_invoice":          "faq_agent",
    "check_invoices":       "faq_agent",

    # Promo agent — pisah dari faq_agent
    "check_voucher":        "promo_agent",
    "check_promo":          "promo_agent",
    "promo_info":           "promo_agent",
    "voucher_info":         "promo_agent",

    # Order agent
    "track_order":          "order_agent",
    "change_order":         "order_agent",
    "delivery_options":     "order_agent",
    "cod_payment":          "order_agent",

    # Product agent
    "check_stock":          "product_agent",

    # Escalation agent
    "get_refund":           "escalation_agent",
    "return_request":       "escalation_agent",
    "seller_complaint":     "escalation_agent",
    "payment_issue":        "escalation_agent",
    "complaint":            "escalation_agent",
    "escalate_to_human":    "escalation_agent",
}

INTENT_LIST = ", ".join(INTENT_TO_AGENT.keys())

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
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

        self.graph = self._build_graph()

    def _classify_intent(self, state: AgentState) -> AgentState:
        prompt = f"""Klasifikasikan intent dari pertanyaan customer berikut.
Pilih SATU intent dari daftar ini:
{INTENT_LIST}

Pertanyaan: {state['query']}

Jawab HANYA dengan nama intent, tidak ada teks lain."""

        response = self.router_llm.invoke(prompt)
        intent   = response.content.strip().lower().replace(" ", "_")

        if intent not in INTENT_TO_AGENT:
            intent = "escalate_to_human"

        agent = INTENT_TO_AGENT[intent]

        return {**state, "intent": intent, "agent": agent}

    def _route_to_agent(self, state: AgentState) -> str:
        return state["agent"]

    def _run_faq_agent(self, state: AgentState) -> AgentState:
        result = self.agents["faq_agent"].run(state["query"])
        return {**state, **result}

    def _run_product_agent(self, state: AgentState) -> AgentState:
        result = self.agents["product_agent"].run(state["query"])
        return {**state, **result}

    def _run_order_agent(self, state: AgentState) -> AgentState:
        result = self.agents["order_agent"].run(state["query"])
        return {**state, **result}

    def _run_escalation_agent(self, state: AgentState) -> AgentState:
        result = self.agents["escalation_agent"].run(state["query"])
        escalate = result.get("confidence") == "low"
        return {**state, **result, "escalate": escalate}

    def _run_promo_agent(self, state: AgentState) -> AgentState:
        result = self.agents["promo_agent"].run(state["query"])
        return {**state, **result}

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        graph = StateGraph(AgentState)

        graph.add_node("classify_intent",  self._classify_intent)
        graph.add_node("faq_agent",        self._run_faq_agent)
        graph.add_node("product_agent",    self._run_product_agent)
        graph.add_node("order_agent",      self._run_order_agent)
        graph.add_node("escalation_agent", self._run_escalation_agent)
        graph.add_node("promo_agent",      self._run_promo_agent)

        # Entry point
        graph.set_entry_point("classify_intent")

        # Conditional routing setelah classify
        graph.add_conditional_edges(
            "classify_intent",
            self._route_to_agent,
            {
                "faq_agent":        "faq_agent",
                "product_agent":    "product_agent",
                "order_agent":      "order_agent",
                "escalation_agent": "escalation_agent",
                "promo_agent":      "promo_agent",
            }
        )

        # Semua agent → END
        graph.add_edge("faq_agent",        END)
        graph.add_edge("product_agent",    END)
        graph.add_edge("order_agent",      END)
        graph.add_edge("escalation_agent", END)
        graph.add_edge("promo_agent",      END)

        return graph.compile()

    @traceable(name="ecommerce-rag-orchestrator", run_type="chain")
    def run(self, query: str) -> dict:
        """Jalankan pipeline lengkap untuk satu query."""
        initial_state = AgentState(
            query=query,
            intent="",
            agent="",
            context=[],
            answer="",
            sources=[],
            confidence="",
            escalate=False,
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
    
    