from langchain_core.tools import tool
from services.product_service import check_stock, get_product_price


@tool
def tool_check_stock(sku: str, size: str = "", color: str = "") -> str:
    """
    Cek stok produk berdasarkan SKU, ukuran, dan warna.
    Gunakan saat customer tanya apakah produk tersedia atau stok habis.
    Args:
        sku: kode produk misal SKU-10001
        size: ukuran misal S, M, L, XL, atau nomor sepatu 39, 40, 41
        color: warna misal Putih, Hitam, Navy
    """
    result = check_stock(sku, size, color)

    if not result["found"]:
        return f"Produk {sku} tidak ditemukan."

    if "reason" in result:
        available = result.get("available_sizes") or result.get("available_colors", [])
        return (
            f"{result['reason']}\n"
            f"Tersedia: {', '.join(available)}" if available else result["reason"]
        )

    if size and color:
        return (
            f"Produk: {result['name']} ({sku})\n"
            f"Ukuran {result['size']} - {result['color']}: "
            f"{result['stock']} pcs — {result['status'].upper()}"
        )

    if size:
        stock_info = ", ".join(
            f"{c}: {q} pcs" for c, q in result["stock"].items()
        )
        return (
            f"Produk: {result['name']} ({sku})\n"
            f"Ukuran {result['size']}: {stock_info}\n"
            f"Status: {result['status'].upper()}"
        )

    return (
        f"Produk: {result['name']} ({sku})\n"
        f"Total stok: {result['total_stock']} pcs — {result['status'].upper()}"
    )


@tool
def tool_get_product_price(sku: str) -> str:
    """
    Cek harga terkini produk berdasarkan SKU.
    Gunakan saat customer tanya harga produk.
    Args:
        sku: kode produk misal SKU-10001
    """
    result = get_product_price(sku)

    if not result["found"]:
        return f"Produk {sku} tidak ditemukan."

    return (
        f"Produk: {result['name']} ({sku})\n"
        f"Harga: Rp{result['price']:,.0f}\n"
        f"Seller: {result['seller']}"
    )