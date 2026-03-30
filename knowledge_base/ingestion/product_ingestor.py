"""
Ingestor for product_catalog collections.
Embedded text: name + category + description + tags
"""

import argparse
from knowledge_base.ingestion.base_ingestor import BaseIngestor

class ProductIngestor(BaseIngestor):
    def __init__(self, qdrant_url, qdrant_api_key ,embeddings=None):
        super().__init__(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name="ecommerce_product_catalog",
            data_path="data/synthetic/product_catalog.json",
            embeddings=embeddings,
        )

    def build_text_for_embedding(self, doc: dict) -> str:
        tags_text = " ".join(doc.get("tags", []))
        specs_text = " ".join(
            f"{k} {v}" for k, v in doc.get("specs", {}).items()
        )
        return (
            f"{doc['name']} {doc['category']} "
            f"{doc['description']} {tags_text} {specs_text}"
        )

    def build_payload(self, doc: dict) -> dict:
        return {
            "doc_id":      doc["doc_id"],
            "collection":  doc["collection"],
            "intent":      doc["intent"],
            "product_id":  doc["product_id"],
            "name":        doc["name"],
            "category":    doc["category"],
            "price":       doc["price"],
            "description": doc["description"],
            "specs":       doc["specs"],
            "seller":      doc["seller"],
            "rating":      doc["rating"],
            "tags":        doc["tags"],
            "language":    doc["language"],
            "updated_at":  doc["updated_at"],
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    ingestor = ProductIngestor(args.url, args.api_key)
    ingestor.ingest(recreate=args.recreate)