---
title: Ecommerce Chat Agent Api
emoji: 🐢
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

# Ecommerce Customer Service Agent

Agentic RAG system untuk customer service e-commerce. Dibangun sebagai proyek portofolio.

## Demo

- **API**: [ecommerce-chat-agent-api](https://huggingface.co/spaces/BadBoyBlack/ecommerce-chat-agent-api)
- **UI**: [ecommerce-chat-agent-ui](https://huggingface.co/spaces/BadBoyBlack/ecommerce-cs-agent-ui)

## Overview

System ini menerima query dari user, mengklasifikasi intentnya, lalu routing ke agent yang sesuai. Setiap agent melakukan hybrid retrieval (dense + sparse) dari knowledge base yang relevan, kemudian generate jawaban menggunakan LLM.

Total knowledge base: 165 dokumen synthetic yang mencakup FAQ, SOP eskalasi, riwayat tiket, promo/voucher, dan katalog produk.

## Arsitektur
```
User Query
  → Intent Classifier (LangGraph router)
  → Agent Router
      ├── FAQ Agent        → ecommerce_faq_policy
      ├── Promo Agent      → ecommerce_promo_voucher  
      ├── Order Agent      → ticket_history + sop_escalation
      ├── Escalation Agent → sop_escalation + ticket_history
      └── Product Agent    → ecommerce_product_catalog
  → Hybrid Retrieval (Dense + BM25 + RRF Fusion + Rerank)
  → LLM Generation
  → Response + sources + confidence + escalate flag
```

## Tech Stack

| Komponen | Tools |
|---|---|
| Orchestrator | LangGraph |
| LLM | OpenRouter (openai/gpt-oss-120b:free) |
| Embedding | BAAI/bge-m3 |
| Vector Store | Qdrant Cloud |
| Sparse Search | BM25 (rank-bm25) |
| API | FastAPI |
| UI | Chainlit |
| Evaluation | RAGAS |
| Tracing | LangSmith |

## Hasil Evaluasi RAGAS

Evaluasi dilakukan dengan 5 sample menggunakan Qwen2.5 sebagai LLM judge (lokal).

| Metric | Score |
|---|---|
| Context Precision | 0.9753 |
| Context Recall | 1.0000 |
| Faithfulness | 0.6683 |
| Answer Relevancy | 0.4195 |


## Struktur Folder
```
├── agents/          # 5 agents + LangGraph orchestrator
├── retrieval/       # dense, sparse, hybrid retriever
├── knowledge_base/  # ingestor per collection
├── api/             # FastAPI endpoint
├── evaluation/      # RAGAS eval pipeline
└── data/synthetic/  # 165 dokumen knowledge base
```

## Setup
```bash
git clone https://github.com/BadBoyBlack/ecommerce-cs-agent
cd ecommerce-cs-agent

pip install -r requirements.txt

cp .env.example .env
# isi .env dengan credentials kamu

# ingest data ke Qdrant
python ingest_all.py

# jalankan API
uvicorn api.main:app --reload
```

## Known Limitations

1. **Answer relevancy rendah (0.42)** — LLM terlalu verbose, perlu prompt yang lebih ketat atau model lebih besar
2. **Faithfulness sedang (0.67)** — LLM kadang keluar dari konteks RAG
3. **Knowledge base kecil** — 25 produk dan 50 tiket, perlu di-expand untuk coverage lebih baik
4. **Reranker sederhana** — keyword overlap, belum pakai CrossEncoder
5. **Tidak ada conversation history** — setiap query diproses independen