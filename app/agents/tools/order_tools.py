from langchain_core.tools import tool
from services.order_service import get_order, get_order_status, get_orders_by_customer

@tool
def tool_get_order(order_id: str) -> str:
    """Ambil detail pesanan berdasarkan order ID."""
    result = get_order(order_id)
    if not result["found"]:
        return f"Pesanan {order_id} tidak ditemukan."

    return (
        f"Order ID: {result['order_id']}\n"
        f"Ticket ID: {result['ticket_id']}\n"
        f"Status: {result['status_label']}\n"
        f"Kategori: {result['category']}\n"
        f"Issue: {result['issue']}\n"
        f"Resolusi: {result['resolution']}\n"
        f"Assigned to: {result['assigned_to']}\n"
        f"Dieskalasi: {'Ya' if result['escalated'] else 'Tidak'}\n"
        f"Waktu Selesai: {result['resolved_in']}"
    )


@tool
def tool_get_order_status(order_id: str) -> str:
    """Cek status pesanan berdasarkan order ID."""
    result = get_order_status(order_id)
    if not result["found"]:
        return f"Pesanan {order_id} tidak ditemukan."

    return (
        f"Order ID: {result['order_id']}\n"
        f"Ticket ID: {result['ticket_id']}\n"
        f"Status: {result['order_status']}\n"
        f"Assigned to: {result['assigned_to']}\n"
        f"Dieskalasi: {'Ya' if result['escalated'] else 'Tidak'}\n"
        f"Waktu Selesai: {result['resolved_in']}"
    )

@tool
def tool_get_orders_by_customer(customer_name: str) -> str:
    """Cari riwayat pesanan berdasarkan nama customer."""
    result = get_orders_by_customer(customer_name)
    if not result["found"]:
        return f"Tidak ada pesanan atas nama {customer_name}. {result.get('reason', '')}"

    return f"Pencarian pesanan atas nama {customer_name}: {result.get('reason', 'Tidak ada data.')}"