"""
Sparse retrieval uses BM25.
BM25 is built from an existing document payload in Qdrant.
"""
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
import re

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

class SparseRetriever:
    def __init__(self, client: QdrantClient):
        self.client = client
        self._index: dict[str, dict] = {}

    def _build_index(self, collection_name: str):
        docs     = []
        offset   = None

        while True:
            results, offset = self.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            docs.extend(results)
            if offset is None:
                break

        corpus     = []
        doc_lookup = []

        for point in docs:
            payload = point.payload
            text = " ".join(filter(None, [
                payload.get("question", ""),
                payload.get("answer", ""),
                payload.get("title", ""),
                payload.get("trigger", ""),
                payload.get("issue", ""),
                payload.get("resolution", ""),
                payload.get("name", ""),
                payload.get("description", ""),
                payload.get("code", ""),
                " ".join(payload.get("tags", [])),
            ]))
            tokens = tokenize(text)
            corpus.append(tokens)
            doc_lookup.append({
                "doc_id":  payload.get("doc_id"),
                "payload": payload,
            })

        self._index[collection_name] = {
            "bm25":       BM25Okapi(corpus),
            "doc_lookup": doc_lookup,
        }
        print(f"BM25 index built: {len(docs)} docs ({collection_name})")


    def preload_all(self, collections: list[str]):
        for col in collections:
            if col not in self._index:
                self._build_index(col)
                
    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
    ) -> list[dict]:

        if collection_name not in self._index:
            self._build_index(collection_name)

        index      = self._index[collection_name]
        bm25       = index["bm25"]
        doc_lookup = index["doc_lookup"]

        tokens = tokenize(query)
        scores = bm25.get_scores(tokens)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [
            {
                "doc_id":  doc_lookup[i]["doc_id"],
                "score":   float(scores[i]),
                "payload": doc_lookup[i]["payload"],
                "source":  "sparse",
            }
            for i in top_indices
            if scores[i] > 0
        ]