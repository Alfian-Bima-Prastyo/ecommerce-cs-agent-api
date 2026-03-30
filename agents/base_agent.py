"""
Base agent — shared logic for all agent.
"""
import os
from langchain_groq import ChatGroq
from retrieval.hybrid_retriever import HybridRetriever

class BaseAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        model: str = None,
        top_k: int = 3,
    ):
        self.retriever = retriever
        self.top_k     = top_k
        self.llm       = ChatGroq(
            model=model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

    def _format_context(self, results: list[dict]) -> str:
        if not results:
            return "Tidak ada konteks relevan ditemukan."

        parts = []
        for i, r in enumerate(results, 1):
            payload = r["payload"]
            content = (
                payload.get("answer") or
                payload.get("resolution") or
                payload.get("description") or
                " ".join(payload.get("steps", [])) or
                payload.get("terms", [""])[0]
            )
            source = payload.get("doc_id", f"doc-{i}")
            parts.append(f"[{i}] ({source})\n{content}")

        return "\n\n".join(parts)

    def _build_prompt(self, query: str, context: str) -> str:
        raise NotImplementedError

    def run(self, query: str, intent_filter: str = None) -> dict:
        """
        Main method: retrieve + generate.
        Returns dict: {answer, sources, confidence}
        """
        results = self.retriever.retrieve(
            query=query,
            agent=self.agent_name,
            top_k=self.top_k,
            intent_filter=intent_filter,
        )

        context = self._format_context(results)
        prompt  = self._build_prompt(query, context)

        response = self.llm.invoke(prompt)
        answer   = response.content.strip()

        top_score  = results[0]["rerank_score"] if results else 0
        confidence = "high" if top_score > 0.06 else "medium" if top_score > 0.03 else "low"

        return {
            "answer":     answer,
            "sources":    [r["doc_id"] for r in results],
            "confidence": confidence,
            "agent":      self.agent_name,
        }