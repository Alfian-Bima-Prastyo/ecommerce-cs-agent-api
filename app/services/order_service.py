"""
Order Service — baca dari Qdrant ecommerce_ticket_history.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from services.cache_service import get_cache
import os

COLLECTION = "ecommerce_ticket_history"
cache = get_cache()

STATUS_LABEL = {
    "open":        "Menunggu Diproses",
    "in_progress": "Sedang Diproses",
    "resolved":    "Selesai",
    "escalated":   "Dieskalasi",
    "closed":      "Ditutup",
}


def _get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )

def get_order(order_id: str) -> dict:
    order_id  = order_id.upper().strip()
    cache_key = f"order:{order_id}"
    cached    = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        client = _get_client()
        results, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="order_id", match=MatchValue(value=order_id))]
            ),
            limit=1,
            with_payload=True,
        )
        if not results:
            return {"found": False, "order_id": order_id, "reason": "Pesanan tidak ditemukan."}

        payload = results[0].payload
        status  = payload.get("status", "")
        result  = {
            "found":        True,
            "order_id":     order_id,
            "ticket_id":    payload.get("ticket_id", ""),
            "status":       status,
            "status_label": STATUS_LABEL.get(status, status),
            "issue":        payload.get("issue", ""),
            "resolution":   payload.get("resolution", ""),
            "assigned_to":  payload.get("assigned_to", ""),
            "escalated":    payload.get("escalated", False),
            "resolved_in":  payload.get("resolved_in", ""),
            "category":     payload.get("category", ""),
        }
        cache.set(cache_key, result)
        return result

    except Exception as e:
        print(f"[WARN] Qdrant error, using fallback for order {order_id}: {e}")
        return {"found": False, "order_id": order_id, "reason": "Pesanan tidak ditemukan."}

    payload = results[0].payload
    status  = payload.get("status", "")
    result  = {
        "found":        True,
        "order_id":     order_id,
        "ticket_id":    payload.get("ticket_id", ""),
        "status":       status,
        "status_label": STATUS_LABEL.get(status, status),
        "issue":        payload.get("issue", ""),
        "resolution":   payload.get("resolution", ""),
        "assigned_to":  payload.get("assigned_to", ""),
        "escalated":    payload.get("escalated", False),
        "resolved_in":  payload.get("resolved_in", ""),
        "category":     payload.get("category", ""),
    }
    cache.set(cache_key, result)
    return result


def get_order_status(order_id: str) -> dict:
    result = get_order(order_id)
    if not result["found"]:
        return result
    return {
        "found":        True,
        "order_id":     result["order_id"],
        "order_status": result["status_label"],
        "ticket_id":    result["ticket_id"],
        "assigned_to":  result["assigned_to"],
        "escalated":    result["escalated"],
        "resolved_in":  result["resolved_in"],
    }


def get_orders_by_customer(customer_name: str) -> dict:
    return {
        "found":         False,
        "customer_name": customer_name,
        "reason":        "Pencarian by nama customer belum didukung di collection ticket_history.",
    }