import json
import time
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

VECTOR_SIZE = 1024

class BaseIngestor:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str,
        data_path: str,
        embeddings=None,
    ):
        self.collection_name = collection_name
        self.data_path       = Path(data_path)

        if embeddings is not None:
            self.embeddings = embeddings
        else:
            print(f"Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
        )
        print(f"Connected to Qdrant: {qdrant_url}")

    def create_collection(self, recreate: bool = False):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            if recreate:
                self.client.delete_collection(self.collection_name)
                print(f"Collection {self.collection_name} dihapus.")
            else:
                print(f"Collection {self.collection_name} sudah ada, skip.")
                return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Collection {self.collection_name} berhasil dibuat.")

    def load_documents(self) -> list[dict]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data.get("documents", [])
        print(f"Loaded {len(docs)} dokumen dari {self.data_path}")
        return docs

    def build_text_for_embedding(self, doc: dict) -> str:
        raise NotImplementedError

    def build_payload(self, doc: dict) -> dict:
        raise NotImplementedError

    def ingest(self, batch_size: int = 5, recreate: bool = False):
        self.create_collection(recreate=recreate)
        docs = self.load_documents()

        print("Generating embeddings (batch)...")
        texts   = [self.build_text_for_embedding(doc) for doc in docs]
        vectors = self.embeddings.embed_documents(texts)  # batch sekaligus
        print(f"Embeddings selesai: {len(vectors)} vektor")

        points  = []
        success = 0
        failed  = 0

        for i, (doc, vector) in enumerate(zip(docs, vectors)):
            try:
                payload = self.build_payload(doc)
                points.append(PointStruct(
                    id=i,
                    vector=vector,
                    payload=payload,
                ))

                if len(points) >= batch_size:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                    success += len(points)
                    print(f"  Uploaded: {success}/{len(docs)}")
                    points = []
                    time.sleep(1)

            except Exception as e:
                print(f"  ERROR doc {doc.get('doc_id')}: {e}")
                failed += 1
                points = []

        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                success += len(points)
            except Exception as e:
                print(f"  ERROR sisa batch: {e}")
                failed += len(points)

        print(f"\nSelesai: {success} berhasil, {failed} gagal")