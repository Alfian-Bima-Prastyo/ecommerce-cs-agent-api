import argparse
from knowledge_base.ingestion.base_ingestor import BaseIngestor

class SOPIngestor(BaseIngestor):
    def __init__(self, qdrant_url, qdrant_api_key ,embeddings=None):
        super().__init__(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name="ecommerce_sop_escalation",
            data_path="data/synthetic/sop_escalation.json",
            embeddings=embeddings,
        )

    def build_text_for_embedding(self, doc: dict) -> str:
        steps_text = " ".join(doc.get("steps", []))
        return f"{doc['title']} {doc['trigger']} {steps_text}"

    def build_payload(self, doc: dict) -> dict:
        return {
            "doc_id":                   doc["doc_id"],
            "collection":               doc["collection"],
            "intent":                   doc["intent"],
            "title":                    doc["title"],
            "trigger":                  doc["trigger"],
            "steps":                    doc["steps"],
            "escalation_level":         doc["escalation_level"],
            "escalation_condition":     doc["escalation_condition"],
            "estimated_resolution_time":doc["estimated_resolution_time"],
            "language":                 doc["language"],
            "updated_at":               doc["updated_at"],
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    ingestor = SOPIngestor(args.url, args.api_key)
    ingestor.ingest(recreate=args.recreate)