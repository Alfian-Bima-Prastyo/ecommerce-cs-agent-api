import pytest
from dotenv import load_dotenv
from unittest.mock import MagicMock
from agents.faq_agent import FAQAgent

load_dotenv()


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "doc_id": "faq-cancel-001",
            "rerank_score": 0.08,
            "payload": {
                "doc_id": "faq-cancel-001",
                "question": "Bagaimana cara membatalkan pesanan?",
                "answer": (
                    "Untuk membatalkan pesanan, buka aplikasi dan masuk ke "
                    "'Pesanan Saya', pilih pesanan yang ingin dibatalkan, "
                    "lalu tekan 'Batalkan Pesanan'. Pastikan status masih "
                    "'Menunggu Konfirmasi' atau 'Diproses'."
                ),
            },
        }
    ]
    return retriever

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    return retriever


def _make_result(doc_id: str, answer: str) -> list:
    return [
        {
            "doc_id": doc_id,
            "rerank_score": 0.08,
            "payload": {"doc_id": doc_id, "answer": answer},
        }
    ]


def test_cancel_order(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "faq-cancel-001",
        "Untuk membatalkan pesanan, buka aplikasi masuk ke menu Pesanan Saya, "
        "pilih pesanan lalu tekan tombol Batalkan Pesanan. "
        "Pastikan status masih Menunggu Konfirmasi atau Diproses.",
    )
    agent  = FAQAgent(retriever=mock_retriever)
    result = agent.run("Bagaimana cara membatalkan pesanan saya?")
    print(f"\nANSWER: {result['answer']}")
    assert result["agent"] == "faq_agent"
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "batalkan", "pesanan", "aplikasi", "menu", "cancel",
        "pembatalan", "membatalkan", "order", "batal",
        "konfirmasi", "diproses", "tombol", "pilih"
    ])



def test_refund_policy(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "faq-refund-001",
        "Proses refund membutuhkan 3 hari kerja untuk saldo wallet "
        "dan 5 hari kerja untuk transfer bank setelah pembatalan dikonfirmasi.",
    )
    agent  = FAQAgent(retriever=mock_retriever)
    result = agent.run("Berapa lama proses refund setelah pembatalan?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "refund", "hari", "kerja", "dana", "kembali", "estimasi"
    ])


def test_delivery_period(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "faq-delivery-001",
        "Estimasi pengiriman ke Jawa Timur adalah 2-3 hari kerja "
        "untuk ekspedisi reguler dan 1 hari untuk same-day.",
    )
    agent  = FAQAgent(retriever=mock_retriever)
    result = agent.run("Berapa lama estimasi pengiriman ke Jawa Timur?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "hari", "pengiriman", "estimasi", "kerja", "ekspedisi"
    ])


def test_payment_options(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "faq-payment-001",
        "Metode pembayaran yang tersedia: transfer bank, kartu kredit, "
        "dompet digital (GoPay, OVO, Dana), dan COD (bayar di tempat).",
    )
    agent  = FAQAgent(retriever=mock_retriever)
    result = agent.run("Metode pembayaran apa saja yang tersedia?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert any(word in result["answer"].lower() for word in [
        "pembayaran", "transfer", "dompet", "kartu", "cod", "bayar"
    ])


def test_create_account(mock_retriever):
    mock_retriever.retrieve.return_value = _make_result(
        "faq-account-001",
        "Untuk membuat akun baru, unduh aplikasi lalu pilih Daftar. "
        "Masukkan nomor HP atau email, buat password, "
        "lalu verifikasi melalui OTP yang dikirim ke nomor Anda.",
    )
    agent  = FAQAgent(retriever=mock_retriever)
    result = agent.run("Bagaimana cara membuat akun baru?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20 
    assert result["agent"] == "faq_agent"
    
def test_unknown_question(mock_retriever):
    mock_retriever.retrieve.return_value = []
    agent  = FAQAgent(retriever=mock_retriever)
    result = agent.run("Apakah platform ini menjual saham?")
    print(f"\nANSWER: {result['answer']}")
    assert result["answer"]
    assert len(result["answer"]) > 20