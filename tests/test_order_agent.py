import pytest
from dotenv import load_dotenv
from unittest.mock import MagicMock
from agents.order_agent import OrderAgent

load_dotenv()


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    return retriever


def _make_result(doc_id: str, content: dict) -> list:
    return [{"doc_id": doc_id, "rerank_score": 0.08, "payload": {"doc_id": doc_id, **content}}]


def test_get_order_detail(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "order-001",
        {"description": "Informasi detail pesanan customer"},
    )
    agent  = OrderAgent(retriever=mock_retriever)
    result = agent.run("Tolong cek detail pesanan ORD-2026-001")
    print(f"\nANSWER: {result['answer']}")
    assert result["agent"] == "order_agent"
    assert result["answer"]
    assert len(result["answer"]) > 20

def test_get_order_status_shipping(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "order-001",
        {"description": "Informasi pengiriman pesanan"},
    )
    agent  = OrderAgent(retriever=mock_retriever)
    result = agent.run("Pesanan ORD-2026-002 sudah sampai mana?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20


def test_get_order_processing(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "order-001",
        {"description": "Status pesanan diproses"},
    )
    agent  = OrderAgent(retriever=mock_retriever)
    result = agent.run("Status pesanan ORD-2026-003 gimana?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20


def test_get_order_cancelled(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "order-001",
        {"description": "Pesanan yang dibatalkan"},
    )
    agent  = OrderAgent(retriever=mock_retriever)
    result = agent.run("Kenapa pesanan ORD-2026-004 dibatalkan?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20

def test_get_orders_by_customer(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "order-001",
        {"description": "Riwayat pesanan customer"},
    )
    agent  = OrderAgent(retriever=mock_retriever)
    result = agent.run("Tampilkan semua pesanan atas nama Budi Santoso")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20


def test_order_not_found(mock_retriever):
    mock_retriever.retrieve.return_value = []
    agent  = OrderAgent(retriever=mock_retriever)
    result = agent.run("Cek pesanan ORD-9999-999")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "tidak ditemukan", "tidak ada", "maaf", "konfirmasi"
    ])