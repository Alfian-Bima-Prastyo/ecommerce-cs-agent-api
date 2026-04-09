import argparse
from knowledge_base.ingestion.base_ingestor import BaseIngestor

class FAQIngestor(BaseIngestor):
    def __init__(self, qdrant_url, qdrant_api_key, embeddings=None):
        super().__init__(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name="ecommerce_faq_policy",
            data_path="data/synthetic/faq_policy.json",
            embeddings=embeddings,
        )

    def build_text_for_embedding(self, doc: dict) -> str:
        return f"{doc['question']} {doc['answer']}"

    def build_payload(self, doc: dict) -> dict:
        return {
            "doc_id":      doc["doc_id"],
            "collection":  doc["collection"],
            "category":    doc["category"],
            "subcategory": doc["subcategory"],
            "intent":      doc["intent"],
            "question":    doc["question"],
            "answer":      doc["answer"],
            "language":    doc["language"],
            "updated_at":  doc["updated_at"],
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    ingestor = FAQIngestor(args.url, args.api_key)
    ingestor.ingest(recreate=args.recreate)