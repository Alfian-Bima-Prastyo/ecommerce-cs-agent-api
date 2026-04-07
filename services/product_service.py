from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from services.cache_service import get_cache
import os

cache = get_cache()
COLLECTION = "ecommerce_product_catalog"


def _get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )

def _find_product_by_sku(sku: str) -> dict | None:
    sku       = sku.upper().strip()
    cache_key = f"product:{sku}"
    cached    = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        client = _get_client()
        results, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="product_id", match=MatchValue(value=sku))]
            ),
            limit=1,
            with_payload=True,
        )
        if not results:
            return None
        payload = results[0].payload
        cache.set(cache_key, payload)
        return payload

    except Exception as e:
        print(f"[WARN] Qdrant error, using fallback for product {sku}: {e}")


COLOR_MAPPING = {
    "black":  "Hitam",
    "white":  "Putih",
    "gray":   "Abu-abu",
    "grey":   "Abu-abu",
    "navy":   "Navy",
    "olive":  "Olive",
    "red":    "Merah",
    "blue":   "Biru",
    "green":  "Hijau",
    "yellow": "Kuning",
}

def check_stock(sku: str, size: str = "", color: str = "") -> dict:
    sku     = sku.upper().strip()
    product = _find_product_by_sku(sku)

    if not product:
        return {"found": False, "sku": sku, "reason": "Produk tidak ditemukan."}

    stock = product.get("stock", {})

    if not size and not color:
        total = sum(
            qty
            for sizes in stock.values()
            for qty in (sizes.values() if isinstance(sizes, dict) else [sizes])
        )
        return {
            "found":       True,
            "sku":         sku,
            "name":        product.get("name"),
            "price":       product.get("price"),
            "total_stock": total,
            "stock":       stock,
            "status":      "tersedia" if total > 0 else "habis",
        }

    size = size.upper().strip()
    if size and size not in stock:
        return {
            "found":           True,
            "sku":             sku,
            "name":            product.get("name"),
            "reason":          f"Ukuran {size} tidak tersedia.",
            "available_sizes": list(stock.keys()),
        }

    size_stock = stock.get(size, {})

    if color:
        color = color.title().strip()
        color = COLOR_MAPPING.get(color.lower(), color)
        qty   = size_stock.get(color) if isinstance(size_stock, dict) else None
        if qty is None:
            return {
                "found":            True,
                "sku":              sku,
                "name":             product.get("name"),
                "reason":           f"Warna {color} tidak tersedia untuk ukuran {size}.",
                "available_colors": list(size_stock.keys()) if isinstance(size_stock, dict) else [],
            }
        return {
            "found":  True,
            "sku":    sku,
            "name":   product.get("name"),
            "size":   size,
            "color":  color,
            "stock":  qty,
            "status": "tersedia" if qty > 0 else "habis",
        }

    return {
        "found":  True,
        "sku":    sku,
        "name":   product.get("name"),
        "size":   size,
        "stock":  size_stock,
        "status": "tersedia" if sum(size_stock.values()) > 0 else "habis",
    }


def get_product_price(sku: str) -> dict:
    sku     = sku.upper().strip()
    product = _find_product_by_sku(sku)

    if not product:
        return {"found": False, "sku": sku, "reason": "Produk tidak ditemukan."}

    return {
        "found":  True,
        "sku":    sku,
        "name":   product.get("name"),
        "price":  product.get("price"),
        "seller": product.get("seller"),
    }