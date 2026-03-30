"""
Ingestor for promo_voucher collections.
Embedded text: code + description + terms
"""

import argparse
from knowledge_base.ingestion.base_ingestor import BaseIngestor

class PromoIngestor(BaseIngestor):
    def __init__(self, qdrant_url, qdrant_api_key ,embeddings=None):
        super().__init__(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name="ecommerce_promo_voucher",
            data_path="data/synthetic/promo_voucher.json",
            embeddings=embeddings,
        )

    def build_text_for_embedding(self, doc: dict) -> str:
        terms_text = " ".join(doc.get("terms", []))
        return f"{doc['code']} {doc['description']} {terms_text}"

    def build_payload(self, doc: dict) -> dict:
        return {
            "doc_id":       doc["doc_id"],
            "collection":   doc["collection"],
            "intent":       doc["intent"],
            "code":         doc["code"],
            "description":  doc["description"],
            "discount":     doc["discount"],
            "min_purchase": doc["min_purchase"],
            "valid_until":  doc["valid_until"],
            "terms":        doc["terms"],
            "language":     doc["language"],
            "updated_at":   doc["updated_at"],
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    ingestor = PromoIngestor(args.url, args.api_key)
    ingestor.ingest(recreate=args.recreate)