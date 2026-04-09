import pytest
from unittest.mock import MagicMock, patch
from agents.promo_agent import PromoAgent
from dotenv import load_dotenv
load_dotenv()

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "doc_id": "promo-001",
            "rerank_score": 0.08,
            "payload": {
                "code": "WELCOME10",
                "description": "Diskon 10% member baru",
                "terms": ["Hanya untuk member baru."],
            },
        }
    ]
    return retriever


@pytest.fixture
def agent(mock_retriever):
    return PromoAgent(retriever=mock_retriever)


def test_valid_voucher(agent):
    result = agent.run("Apakah voucher WELCOME10 masih bisa dipakai?")
    assert result["agent"] == "promo_agent"
    assert result["answer"]
    assert "WELCOME10" in result["answer"] or "valid" in result["answer"].lower()


def test_invalid_voucher_expired(agent):
    result = agent.run("Kenapa voucher FLASHSALE25 tidak bisa dipakai?")
    assert result["answer"]
    assert len(result["answer"]) > 20
    
def test_voucher_wrong_category(agent):
    result = agent.run(
        "Saya mau pakai voucher SAVE50 untuk beli baju, "
        "total belanja Rp600.000. Bisa tidak?"
    )
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "elektronik", "kategori", "tidak valid", "tidak bisa", 
        "tidak dapat", "hanya berlaku", "khusus", "fashion"
    ])

def test_active_promos(agent):
    result = agent.run("Promo apa yang sedang berlaku sekarang?")
    assert result["answer"]
    assert len(result["answer"]) > 20

def test_voucher_not_found(agent):
    result = agent.run("Cek voucher TIDAKADA123 dong")
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "tidak ditemukan", "tidak valid", "tidak ada", 
        "tidak tersedia", "tidakada123", "maaf"
    ])


def test_min_purchase_not_met(agent):
    result = agent.run(
        "Saya mau pakai SAVE50 tapi total belanja saya Rp200.000, bisa tidak?"
    )
    assert result["answer"]
    assert len(result["answer"]) > 20