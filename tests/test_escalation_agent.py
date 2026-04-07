import pytest
from dotenv import load_dotenv
from unittest.mock import MagicMock
from agents.escalation_agent import EscalationAgent

load_dotenv()


@pytest.fixture

def mock_retriever():
    retriever = MagicMock()
    return retriever


def _make_result(doc_id: str, content: dict) -> list:
    return [{"doc_id": doc_id, "rerank_score": 0.08, "payload": {"doc_id": doc_id, **content}}]


def test_create_ticket_complaint(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "esc-001",
        {"description": "Informasi proses eskalasi dan komplain"},
    )
    agent  = EscalationAgent(retriever=mock_retriever)
    result = agent.run("Saya mau komplain, produk yang saya terima rusak. Nama saya Budi, order ORD-2026-001")
    print(f"\nANSWER: {result['answer']}")
    assert result["agent"] == "escalation_agent"
    assert result["answer"]
    assert len(result["answer"]) > 20


def test_get_ticket_detail(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "esc-001",
        {"description": "Detail tiket eskalasi"},
    )
    agent  = EscalationAgent(retriever=mock_retriever)
    result = agent.run("Cek status tiket saya TKT-2026-001")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20


def test_get_ticket_not_found(mock_retriever):
    mock_retriever.retrieve.return_value = []
    agent  = EscalationAgent(retriever=mock_retriever)
    result = agent.run("Cek tiket TKT-9999-999")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "tidak ditemukan", "tidak ada", "maaf", "konfirmasi"
    ])


def test_get_tickets_by_customer(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "esc-001",
        {"description": "Riwayat tiket customer"},
    )
    agent  = EscalationAgent(retriever=mock_retriever)
    result = agent.run("Tampilkan semua tiket atas nama Siti Rahayu")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20


def test_escalate_shipping_issue(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "esc-001",
        {"description": "Masalah pengiriman yang perlu dieskalasi"},
    )
    agent  = EscalationAgent(retriever=mock_retriever)
    result = agent.run("Paket saya ORD-2026-002 sudah seminggu tidak sampai, tolong bantu eskalasi")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20


def test_general_escalation_info(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "esc-001",
        {"description": "Informasi umum proses eskalasi dan waktu respon"},
    )
    agent  = EscalationAgent(retriever=mock_retriever)
    result = agent.run("Berapa lama proses eskalasi ke human agent?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20