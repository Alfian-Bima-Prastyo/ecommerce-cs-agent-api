import pytest
from dotenv import load_dotenv
from unittest.mock import MagicMock
from agents.product_agent import ProductAgent

load_dotenv()


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    return retriever


def _make_result(doc_id: str, content: dict) -> list:
    return [{"doc_id": doc_id, "rerank_score": 0.08, "payload": {"doc_id": doc_id, **content}}]


def test_check_stock_available(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "prod-001",
        {"name": "Kaos Polos Premium", "description": "Kaos cotton combed 30s"},
    )
    agent  = ProductAgent(retriever=mock_retriever)
    result = agent.run("Apakah SKU-10001 ukuran M warna Hitam masih tersedia?")
    print(f"\nANSWER: {result['answer']}")
    assert result["agent"] == "product_agent"
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "tersedia", "stok", "hitam", "m", "sku-10001", "pcs"
    ])


def test_check_stock_habis(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "prod-001",
        {"name": "Kaos Polos Premium", "description": "Kaos cotton combed 30s"},
    )
    agent  = ProductAgent(retriever=mock_retriever)
    result = agent.run("Stok SKU-10001 ukuran XL warna Olive masih ada?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "habis", "tidak tersedia", "kosong", "0", "stok"
    ])


def test_get_price(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "prod-001",
        {"name": "Kaos Polos Premium", "price": 95000},
    )
    agent  = ProductAgent(retriever=mock_retriever)
    result = agent.run("Berapa harga SKU-10001?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20


def test_product_search(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "prod-001",
        {
            "name": "Kaos Polos Premium Cotton Combed 30s",
            "description": "Kaos premium bahan cotton combed 30s",
            "price": 95000,
            "specs": {"bahan": "Cotton Combed 30s", "ukuran_tersedia": "S, M, L, XL, XXL"},
        },
    )
    agent  = ProductAgent(retriever=mock_retriever)
    result = agent.run("Rekomendasikan kaos pria yang bagus")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "kaos", "cotton", "premium", "pria", "bahan"
    ])


def test_product_not_found(mock_retriever):
    mock_retriever.retrieve.return_value = []
    agent  = ProductAgent(retriever=mock_retriever)
    result = agent.run("Cek stok SKU-99999")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20


def test_compare_products(mock_retriever):
    mock_retriever.retrieve.return_value = [
        {
            "doc_id": "prod-001",
            "rerank_score": 0.08,
            "payload": {
                "doc_id": "prod-001",
                "name": "Kaos Polos Premium Cotton Combed 30s",
                "price": 95000,
                "specs": {"bahan": "Cotton Combed 30s", "gramasi": "180 gsm"},
                "rating": 4.7,
            },
        },
        {
            "doc_id": "prod-002",
            "rerank_score": 0.07,
            "payload": {
                "doc_id": "prod-002",
                "name": "Kemeja Flannel Pria Lengan Panjang",
                "price": 185000,
                "specs": {"bahan": "Flannel", "lengan": "Panjang"},
                "rating": 4.5,
            },
        },
    ]
    agent  = ProductAgent(retriever=mock_retriever)
    result = agent.run("Bandingkan kaos cotton combed dengan kemeja flannel")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 50