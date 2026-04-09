import argparse
from knowledge_base.ingestion.base_ingestor import BaseIngestor
from qdrant_client.models import PayloadSchemaType

class TicketIngestor(BaseIngestor):
    def __init__(self, qdrant_url, qdrant_api_key ,embeddings=None):
        super().__init__(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name="ecommerce_ticket_history",
            data_path="data/synthetic/ticket_history.json",
            embeddings=embeddings,
        )

    def build_text_for_embedding(self, doc: dict) -> str:
        return f"{doc['issue']} {doc['resolution']} {doc['root_cause']}"

    def build_payload(self, doc: dict) -> dict:
        intents           = doc.get("intents", [])
        primary_intent    = next(
            (i["intent"] for i in intents if i.get("primary")), None
        )
        secondary_intents = [
            i["intent"] for i in intents if not i.get("primary")
        ]

        return {
            "doc_id":            doc["doc_id"],
            "collection":        doc["collection"],
            "primary_intent":    primary_intent,
            "secondary_intents": secondary_intents,
            "all_intents":       [i["intent"] for i in intents],
            "category":          doc["category"],
            "issue":             doc["issue"],
            "resolution":        doc["resolution"],
            "root_cause":        doc["root_cause"],
            "escalated":         doc["escalated"],
            "resolved_in":       doc["resolved_in"],
            "ticket_id":         doc.get("ticket_id", ""),
            "order_id":          doc.get("order_id", ""),
            "assigned_to":       doc.get("assigned_to", ""),
            "status":            doc.get("status", ""),
            "language":          doc["language"],
            "updated_at":        doc["updated_at"],
        }
    
    def create_payload_indexes(self):
        for field in ["order_id", "ticket_id", "customer_name"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    ingestor = TicketIngestor(args.url, args.api_key)
    ingestor.ingest(recreate=args.recreate)