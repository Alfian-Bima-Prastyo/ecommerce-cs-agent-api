from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
import os
import uuid
from services.cache_service import get_cache

cache = get_cache()
COLLECTION = "ecommerce_ticket_history"

CATEGORY_LABEL = {
    "complaint": "Komplain",
    "shipping":  "Pengiriman",
    "product":   "Produk",
    "payment":   "Pembayaran",
    "other":     "Lainnya",
}

PRIORITY_LABEL = {
    "low":    "Rendah",
    "medium": "Sedang",
    "high":   "Tinggi",
}

STATUS_LABEL = {
    "open":        "Menunggu Diproses",
    "in_progress": "Sedang Diproses",
    "resolved":    "Selesai",
    "closed":      "Ditutup",
}


def _get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )


def create_ticket(customer_name: str, category: str, description: str, order_id: str = None, priority: str = "medium") -> dict:
    client    = _get_client()
    ticket_id = f"TKT-{datetime.now().strftime('%Y')}-{str(uuid.uuid4())[:6].upper()}"

    payload = {
        "ticket_id":     ticket_id,
        "order_id":      order_id or "",
        "customer_name": customer_name,
        "category":      category,
        "priority":      priority,
        "status":        "open",
        "description":   description,
        "resolution":    "",
        "assigned_to":   "",
        "created_at":    datetime.now().strftime("%Y-%m-%d"),
        "resolved_at":   "",
        "collection":    COLLECTION,
    }

    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.0] * 1024,
            payload=payload,
        )],
    )

    return {
        "success":        True,
        "ticket_id":      ticket_id,
        "customer_name":  customer_name,
        "category_label": CATEGORY_LABEL.get(category, category),
        "priority_label": PRIORITY_LABEL.get(priority, priority),
        "status_label":   STATUS_LABEL["open"],
        "message":        f"Tiket {ticket_id} berhasil dibuat. Tim kami akan menghubungi Anda dalam 1x24 jam.",
    }


def get_ticket(ticket_id: str) -> dict:
    ticket_id = ticket_id.upper().strip()
    cache_key = f"ticket:{ticket_id}"
    cached    = cache.get(cache_key)
    if cached is not None:
        return cached

    client = _get_client()
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="ticket_id", match=MatchValue(value=ticket_id))]
        ),
        limit=1,
        with_payload=True,
    )
    if not results:
        return {"found": False, "ticket_id": ticket_id, "reason": "Tiket tidak ditemukan."}

    p = results[0].payload
    result = {
        "found":          True,
        "ticket_id":      p.get("ticket_id"),
        "order_id":       p.get("order_id"),
        "customer_name":  p.get("customer_name", ""),
        "category_label": CATEGORY_LABEL.get(p.get("category", ""), p.get("category", "")),
        "priority_label": PRIORITY_LABEL.get(p.get("priority", ""), p.get("priority", "")),
        "status":         p.get("status", ""),
        "status_label":   STATUS_LABEL.get(p.get("status", ""), p.get("status", "")),
        "description":    p.get("description", p.get("issue", "")),
        "resolution":     p.get("resolution", "") or "Masih dalam proses.",
        "assigned_to":    p.get("assigned_to", ""),
        "created_at":     p.get("created_at", p.get("updated_at", "")),
        "resolved_at":    p.get("resolved_at", ""),
    }
    cache.set(cache_key, result)
    return result


def get_tickets_by_customer(customer_name: str) -> dict:
    cache_key = f"tickets_by_customer:{customer_name.strip().lower()}"
    cached    = cache.get(cache_key)
    if cached is not None:
        return cached

    client = _get_client()
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="customer_name", match=MatchValue(value=customer_name.strip()))]
        ),
        limit=10,
        with_payload=True,
    )
    if not results:
        return {"found": False, "customer_name": customer_name, "reason": "Tidak ada tiket ditemukan."}

    tickets = [
        {
            "ticket_id":      p.payload.get("ticket_id"),
            "category_label": CATEGORY_LABEL.get(p.payload.get("category", ""), p.payload.get("category", "")),
            "status_label":   STATUS_LABEL.get(p.payload.get("status", ""), p.payload.get("status", "")),
            "created_at":     p.payload.get("created_at", p.payload.get("updated_at", "")),
        }
        for p in results
    ]
    result = {"found": True, "customer_name": customer_name, "tickets": tickets, "total_tickets": len(tickets)}
    cache.set(cache_key, result)
    return result