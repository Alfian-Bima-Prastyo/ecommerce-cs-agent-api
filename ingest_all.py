import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from knowledge_base.ingestion.faq_ingestor     import FAQIngestor
from knowledge_base.ingestion.sop_ingestor     import SOPIngestor
from knowledge_base.ingestion.ticket_ingestor  import TicketIngestor
from knowledge_base.ingestion.promo_ingestor   import PromoIngestor
from knowledge_base.ingestion.product_ingestor import ProductIngestor

load_dotenv()
URL     = os.getenv("QDRANT_URL")
API_KEY = os.getenv("QDRANT_API_KEY")

# Load SEKALI, share ke semua ingestor
print("Loading embedding model (sekali)...")
shared_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Model loaded.")

ingestors = [
    SOPIngestor(URL, API_KEY, embeddings=shared_embeddings),
    FAQIngestor(URL, API_KEY, embeddings=shared_embeddings),
    TicketIngestor(URL, API_KEY, embeddings=shared_embeddings),
    PromoIngestor(URL, API_KEY, embeddings=shared_embeddings),
    ProductIngestor(URL, API_KEY, embeddings=shared_embeddings),
]

for ingestor in ingestors:
    print(f"\n{'='*50}")
    print(f"Ingesting: {ingestor.collection_name}")
    print(f"{'='*50}")
    ingestor.ingest(recreate=False)