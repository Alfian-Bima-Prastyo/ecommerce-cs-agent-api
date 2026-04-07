from langchain_core.tools import tool
from services.escalation_service import create_ticket, get_ticket, get_tickets_by_customer


@tool
def tool_create_ticket(customer_name: str, category: str, description: str, order_id: str = "", priority: str = "medium") -> str:
    """
    Buat tiket eskalasi baru ke human agent.
    category: complaint, shipping, product, payment, other.
    priority: low, medium, high.
    Gunakan jika customer butuh bantuan yang tidak bisa diselesaikan agent.
    """
    result = create_ticket(
        customer_name=customer_name,
        category=category,
        description=description,
        order_id=order_id or None,
        priority=priority,
    )
    return (
        f"Tiket berhasil dibuat!\n"
        f"Ticket ID: {result['ticket_id']}\n"
        f"Kategori: {result['category_label']}\n"
        f"Prioritas: {result['priority_label']}\n"
        f"Status: {result['status_label']}\n"
        f"{result['message']}"
    )


@tool
def tool_get_ticket(ticket_id: str) -> str:
    """Ambil detail tiket eskalasi berdasarkan ticket ID. Gunakan jika customer menyebut nomor tiket."""
    result = get_ticket(ticket_id)
    if not result["found"]:
        return f"Tiket {ticket_id} tidak ditemukan."

    resolution = result["resolution"] or "Masih dalam proses."
    resolved_at = result["resolved_at"] or "-"

    return (
        f"Ticket ID: {result['ticket_id']}\n"
        f"Customer: {result['customer_name']}\n"
        f"Kategori: {result['category_label']}\n"
        f"Prioritas: {result['priority_label']}\n"
        f"Status: {result['status_label']}\n"
        f"Deskripsi: {result['description']}\n"
        f"Resolusi: {resolution}\n"
        f"Dibuat: {result['created_at']} | Selesai: {resolved_at}"
    )


@tool
def tool_get_tickets_by_customer(customer_name: str) -> str:
    """Cari semua tiket eskalasi berdasarkan nama customer."""
    result = get_tickets_by_customer(customer_name)
    if not result["found"]:
        return f"Tidak ada tiket atas nama {customer_name}."

    tickets_str = "\n".join(
        f"- {t['ticket_id']} | {t['category_label']} | {t['status_label']} | {t['created_at']}"
        for t in result["tickets"]
    )
    return (
        f"Tiket atas nama {result['customer_name']} ({result['total_tickets']} tiket):\n"
        f"{tickets_str}"
    )