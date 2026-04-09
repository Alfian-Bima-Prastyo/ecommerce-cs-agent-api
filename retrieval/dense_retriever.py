from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings

class DenseRetriever:
    def __init__(self, client: QdrantClient, embeddings: HuggingFaceEmbeddings):
        self.client     = client
        self.embeddings = embeddings

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        intent_filter: str = None,
    ) -> list[dict]:

        query_vector = self.embeddings.embed_query(query)

        query_filter = None
        if intent_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="intent",
                    match=MatchValue(value=intent_filter)
                )]
            )

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        return [
            {
                "doc_id":  hit.payload.get("doc_id"),
                "score":   hit.score,
                "payload": hit.payload,
                "source":  "dense",
            }
            for hit in results
        ]