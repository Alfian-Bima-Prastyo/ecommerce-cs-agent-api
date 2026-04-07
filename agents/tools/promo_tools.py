from langchain_core.tools import tool
from services.voucher_service import validate_voucher, get_active_promos, check_voucher_expiry, apply_voucher

@tool
def tool_validate_voucher(code: str, purchase_amount: str = "0", category: str = "") -> str:
    """
    Validasi apakah voucher bisa digunakan customer.
    Gunakan saat customer tanya voucher tidak bisa dipakai atau ingin cek status voucher.
    Args:
        code: kode voucher misal SAVE50
        purchase_amount: total belanja dalam rupiah sebagai string angka, "0" jika tidak diketahui
        category: kategori produk misal elektronik, fashion
    """
    try:
        amount = float(str(purchase_amount).replace(",", "").replace("Rp", "").strip())
    except (ValueError, TypeError):
        amount = 0.0

    result = validate_voucher(code, amount, category)
    if result["valid"]:
        return (
            f"Voucher {code} VALID.\n"
            f"Deskripsi: {result['description']}\n"
            f"Diskon: {result['discount_type']} {result['discount_value']}"
            + (f" = Rp{result['discount_amount']:,.0f}" if result['discount_amount'] else "") + "\n"
            f"Min. pembelian: Rp{result['min_purchase']:,.0f}\n"
            f"Berlaku hingga: {result['valid_until']}\n"
            f"Kuota tersisa: {result['quota_remaining']}\n"
            f"Syarat: {'; '.join(result['terms'])}"
        )
    return f"Voucher {code} TIDAK VALID. Alasan: {result['reason']}"


@tool
def tool_check_voucher_expiry(code: str) -> str:
    """
    Cek tanggal expired voucher.
    Gunakan saat customer tanya apakah voucher masih berlaku atau sudah expired.
    Args:
        code: kode voucher
    """
    result = check_voucher_expiry(code)
    if not result["found"]:
        return f"Voucher {code} tidak ditemukan."
    return (
        f"Voucher {code}: status {result['status'].upper()}\n"
        f"Berlaku hingga: {result['valid_until']}\n"
        f"Sisa hari: {result['days_left']} hari"
    )

@tool
def tool_get_active_promos() -> str:
    """
    Ambil daftar promo yang sedang aktif.
    Gunakan saat customer tanya promo apa yang sedang berjalan.
    """
    promos = get_active_promos()
    if not promos:
        return "Tidak ada promo aktif saat ini."
    parts = []
    for p in promos:
        discount = p.get("discount", {})
        if discount.get("type") == "persen":
            diskon_str = f"Diskon {discount['value']}%"
        elif discount.get("type") == "nominal":
            diskon_str = f"Potongan Rp{discount['value']:,}"
        else:
            diskon_str = "Gratis ongkir"
        parts.append(
            f"- {p['code']} — {diskon_str}\n"
            f"  {p['description']}\n"
            f"  Berlaku hingga: {p['valid_until']}"
        )
    return "Promo aktif saat ini:\n" + "\n".join(parts)

@tool
def tool_apply_voucher(code: str, cart_total: str, category: str = "") -> str:
    """
    Hitung harga akhir setelah voucher diterapkan.
    Gunakan jika customer ingin tahu berapa harga setelah pakai voucher.
    cart_total adalah total belanja dalam rupiah, kirim sebagai string angka.
    Contoh: cart_total="500000" untuk belanja Rp500.000
    category opsional.
    """
    try:
        total = float(str(cart_total).replace(",", "").replace("Rp", "").strip())
    except (ValueError, TypeError):
        return "Maaf, format total belanja tidak valid. Contoh yang benar: 500000"

    result = apply_voucher(code, total, category)

    if not result["success"]:
        return f"Voucher {code} tidak dapat diterapkan: {result['reason']}"

    return (
        f"Voucher {result['code']} berhasil diterapkan!\n"
        f"Total awal: Rp{result['original_total']:,.0f}\n"
        f"Diskon ({result['discount_type']} {result['discount_value']}{'%' if result['discount_type'] == 'persen' else ''}): "
        f"-Rp{result['discount_amount']:,.0f}\n"
        f"Total akhir: Rp{result['final_total']:,.0f}\n"
        f"Hemat: {result['savings_pct']}%\n"
        f"Kuota tersisa: {result['quota_remaining']}"
    )