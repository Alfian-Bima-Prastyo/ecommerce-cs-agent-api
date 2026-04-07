# create_indexes.py
from dotenv import load_dotenv; load_dotenv()
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60,
)

indexes = [
    ("ecommerce_product_catalog", "product_id",    PayloadSchemaType.KEYWORD),
    ("ecommerce_promo_voucher",   "code",           PayloadSchemaType.KEYWORD),
    ("ecommerce_promo_voucher",   "is_active",      PayloadSchemaType.BOOL),     # ← ganti ke BOOL
    ("ecommerce_ticket_history",  "order_id",       PayloadSchemaType.KEYWORD),
    ("ecommerce_ticket_history",  "ticket_id",      PayloadSchemaType.KEYWORD),
    ("ecommerce_ticket_history",  "customer_name",  PayloadSchemaType.KEYWORD),
]

for collection, field, schema in indexes:
    print(f"Creating index: {collection}.{field} ({schema})")
    client.create_payload_index(
        collection_name=collection,
        field_name=field,
        field_schema=schema,
    )
    print(f"  OK")