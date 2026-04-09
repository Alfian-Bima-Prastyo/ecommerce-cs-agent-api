from datetime import date
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from services.cache_service import get_cache
import os

cache = get_cache()
COLLECTION = "ecommerce_promo_voucher"


def _get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )


def _find_voucher_by_code(code: str) -> dict | None:
    code      = code.upper().strip()
    cache_key = f"voucher:{code}"
    cached    = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        client = _get_client()
        results, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="code", match=MatchValue(value=code))]
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
        print(f"[WARN] Qdrant error, using fallback for voucher {code}: {e}")
        


def validate_voucher(code: str, purchase_amount: float = 0.0, category: str = "") -> dict:
    code    = code.upper().strip()
    voucher = _find_voucher_by_code(code)

    if not voucher:
        return {"valid": False, "code": code, "reason": "Voucher tidak ditemukan."}

    if not voucher.get("is_active", True):
        return {"valid": False, "code": code, "reason": "Voucher sudah tidak aktif."}

    valid_until = voucher.get("valid_until", "")
    if valid_until and date.fromisoformat(valid_until) < date.today():
        return {"valid": False, "code": code, "reason": f"Voucher expired pada {valid_until}."}

    used  = voucher.get("used", 0)
    quota = voucher.get("quota", 0)
    if quota and used >= quota:
        return {"valid": False, "code": code, "reason": "Kuota voucher sudah habis."}

    min_purchase = voucher.get("min_purchase", 0)
    if purchase_amount and purchase_amount < min_purchase:
        return {
            "valid":  False,
            "code":   code,
            "reason": f"Minimum pembelian Rp{min_purchase:,.0f}. Belanja Anda Rp{purchase_amount:,.0f}.",
        }

    discount = voucher.get("discount", {})
    discount_amount = 0.0
    if purchase_amount:
        if discount.get("type") == "persen":
            discount_amount = purchase_amount * discount.get("value", 0) / 100
        elif discount.get("type") == "nominal":
            discount_amount = float(discount.get("value", 0))

    return {
        "valid":           True,
        "code":            code,
        "description":     voucher.get("description", ""),
        "discount_type":   discount.get("type", ""),
        "discount_value":  discount.get("value", 0),
        "discount_amount": discount_amount,
        "min_purchase":    min_purchase,
        "valid_until":     valid_until,
        "quota_remaining": quota - used,
        "terms":           voucher.get("terms", []),
    }


def get_active_promos() -> list:
    cache_key = "promos:active"
    cached    = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        client = _get_client()
        results, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="is_active", match=MatchValue(value=True))]
            ),
            limit=20,
            with_payload=True,
        )
        promos = [
            {
                "code":        r.payload.get("code"),
                "description": r.payload.get("description"),
                "valid_until": r.payload.get("valid_until"),
                "discount":    r.payload.get("discount"),
            }
            for r in results
        ]
        cache.set(cache_key, promos)
        return promos

    except Exception as e:
        print(f"[WARN] Qdrant error, using fallback for active promos: {e}")
        return list(FALLBACK_VOUCHERS.values())


def check_voucher_expiry(code: str) -> dict:
    code    = code.upper().strip()
    voucher = _find_voucher_by_code(code)

    if not voucher:
        return {"found": False, "code": code}

    today     = date.today()
    expiry    = date.fromisoformat(voucher.get("valid_until", "2099-12-31"))
    days_left = (expiry - today).days

    return {
        "found":       True,
        "code":        code,
        "valid_until": voucher.get("valid_until"),
        "is_expired":  days_left < 0,
        "days_left":   max(0, days_left),
        "status":      "expired" if days_left < 0 else ("segera expired" if days_left <= 7 else "aktif"),
    }

def apply_voucher(code: str, cart_total: float, category: str = "") -> dict:
    validation = validate_voucher(code, cart_total, category)

    if not validation["valid"]:
        return {
            "success":       False,
            "code":          code,
            "reason":        validation["reason"],
            "original_total": cart_total,
            "discount_amount": 0.0,
            "final_total":   cart_total,
        }

    discount      = validation.get("discount", {})
    discount_type = validation.get("discount_type", "")
    discount_val  = validation.get("discount_value", 0)

    if discount_type == "persen":
        discount_amount = cart_total * discount_val / 100
    elif discount_type == "nominal":
        discount_amount = float(discount_val)
    elif discount_type == "ongkir":
        discount_amount = 0.0  # ongkir di-handle terpisah
    else:
        discount_amount = 0.0

    discount_amount = min(discount_amount, cart_total)
    final_total     = cart_total - discount_amount

    return {
        "success":          True,
        "code":             code,
        "description":      validation.get("description", ""),
        "discount_type":    discount_type,
        "discount_value":   discount_val,
        "discount_amount":  discount_amount,
        "original_total":   cart_total,
        "final_total":      final_total,
        "quota_remaining":  validation.get("quota_remaining", 0),
        "valid_until":      validation.get("valid_until", ""),
        "savings_pct":      round(discount_amount / cart_total * 100, 1) if cart_total > 0 else 0,
    }