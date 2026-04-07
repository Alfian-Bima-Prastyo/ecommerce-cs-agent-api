import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from retrieval.hybrid_retriever import HybridRetriever, AGENT_COLLECTIONS
from agents.orchestrator import Orchestrator
from evaluation.eval_dataset import build_eval_dataset


def run_evaluation():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        model_kwargs={"device": "cpu"},
    )

    print("Connecting to Qdrant...")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )

    print("Building BM25 indexes...")
    retriever = HybridRetriever(client, embeddings)
    retriever.sparse.preload_all(AGENT_COLLECTIONS["all"])

    print("Initializing orchestrator...")
    orchestrator = Orchestrator(retriever)

    print("Building eval dataset...")
    eval_samples = build_eval_dataset()
    print(f"Total samples: {len(eval_samples)}")

    print("\nRunning pipeline on eval samples...")
    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    for i, sample in enumerate(eval_samples, 1):
        print(f"  [{i}/{len(eval_samples)}] {sample['question'][:60]}...")
        try:
            result = orchestrator.run(sample["question"])
            questions.append(sample["question"])
            answers.append(result["answer"])
            contexts_list.append(sample["contexts"])
            ground_truths.append(sample["ground_truth"])
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    ragas_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })

    ragas_llm = LangchainLLMWrapper(
        ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
    )
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=RunConfig(timeout=120, max_workers=1),
    )

    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    print(results)

    df = results.to_pandas()

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n── Aggregate Scores ──")
    for name in metric_names:
        if name in df.columns:
            print(f"  {name:25s}: {df[name].mean():.4f}")

    out_path = Path("evaluation/results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_evaluation()