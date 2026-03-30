"""
Hybrid retrieval: Dense + Sparse + RRF Fusion + Re-ranker.
"""
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from retrieval.dense_retriever  import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever

AGENT_COLLECTIONS = {
    "faq_agent":       ["ecommerce_faq_policy"],
    "product_agent":   ["ecommerce_product_catalog"],
    "order_agent":     ["ecommerce_ticket_history", "ecommerce_sop_escalation"],
    "escalation_agent":["ecommerce_sop_escalation", "ecommerce_ticket_history"],
    "promo_agent":     ["ecommerce_promo_voucher"],
    "all":             [
        "ecommerce_faq_policy",
        "ecommerce_sop_escalation",
        "ecommerce_ticket_history",
        "ecommerce_promo_voucher",
        "ecommerce_product_catalog",
    ],
}

class HybridRetriever:
    def __init__(self, client: QdrantClient, embeddings: HuggingFaceEmbeddings):
        self.dense  = DenseRetriever(client, embeddings)
        self.sparse = SparseRetriever(client)

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        scores = {} 
        docs   = {}

        for rank, result in enumerate(dense_results):
            doc_id = result["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            docs[doc_id]   = result["payload"]

        for rank, result in enumerate(sparse_results):
            doc_id = result["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            docs[doc_id]   = result["payload"]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "doc_id":    doc_id,
                "rrf_score": score,
                "payload":   docs[doc_id],
                "source":    "hybrid",
            }
            for doc_id, score in ranked
        ]

    def _simple_rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int,
    ) -> list[dict]:
        query_tokens = set(query.lower().split())

        for result in results:
            payload = result["payload"]
            text = " ".join(filter(None, [
                payload.get("question", ""),
                payload.get("answer", ""),
                payload.get("title", ""),
                payload.get("issue", ""),
                payload.get("name", ""),
                payload.get("description", ""),
            ])).lower()

            doc_tokens    = set(text.split())
            overlap       = len(query_tokens & doc_tokens)
            result["rerank_score"] = result["rrf_score"] + (overlap * 0.01)

        return sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True
        )[:top_k]

    def retrieve(
        self,
        query: str,
        agent: str = "all",
        top_k: int = 5,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        intent_filter: str = None,
    ) -> list[dict]:

        collections = AGENT_COLLECTIONS.get(agent, AGENT_COLLECTIONS["all"])

        all_dense  = []
        all_sparse = []

        for collection in collections:
            all_dense.extend(self.dense.retrieve(
                query=query,
                collection_name=collection,
                top_k=dense_top_k,
                intent_filter=intent_filter,
            ))
            all_sparse.extend(self.sparse.retrieve(
                query=query,
                collection_name=collection,
                top_k=sparse_top_k,
            ))

        fused = self._rrf_fusion(all_dense, all_sparse)

        reranked = self._simple_rerank(query, fused, top_k)

        return reranked