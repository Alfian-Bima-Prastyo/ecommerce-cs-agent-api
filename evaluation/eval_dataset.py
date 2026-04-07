import json
import random
from pathlib import Path

DATA_DIR = Path("data/synthetic")

def load_faq_samples(n: int = 5) -> list[dict]:
    with open(DATA_DIR / "faq_policy.json", encoding="utf-8") as f:
        docs = json.load(f)["documents"]
    samples = random.sample(docs, min(n, len(docs)))
    return [
        {
            "question":     d["question"],
            "ground_truth": d["answer"],
            "contexts":     [d["answer"]],
            "agent":        "faq_agent",
            "intent":       d["intent"],
        }
        for d in samples
    ]

def load_product_samples(n: int = 3) -> list[dict]:
    with open(DATA_DIR / "product_catalog.json", encoding="utf-8") as f:
        docs = json.load(f)["documents"]
    samples = random.sample(docs, min(n, len(docs)))
    return [
        {
            "question":     f"Apakah {d['name']} tersedia? Apa spesifikasinya?",
            "ground_truth": d["description"],
            "contexts":     [d["description"]],
            "agent":        "product_agent",
            "intent":       d["intent"],
        }
        for d in samples
    ]

def load_promo_samples(n: int = 3) -> list[dict]:
    with open(DATA_DIR / "promo_voucher.json", encoding="utf-8") as f:
        docs = json.load(f)["documents"]
    samples = random.sample(docs, min(n, len(docs)))
    return [
        {
            "question":     f"Bagaimana cara menggunakan voucher {d['code']}?",
            "ground_truth": d["description"] + " " + " ".join(d.get("terms", [])),
            "contexts":     [d["description"], *d.get("terms", [])],
            "agent":        "promo_agent",
            "intent":       d["intent"],
        }
        for d in samples
    ]

def load_order_samples(n: int = 3) -> list[dict]:
    with open(DATA_DIR / "ticket_history.json", encoding="utf-8") as f:
        docs = json.load(f)["documents"]
        
    order_intents = {"track_order", "change_order", "delivery_options", "cod_payment"}
    filtered = [
        d for d in docs
        if any(
            i["intent"] in order_intents and i["primary"]
            for i in d.get("intents", [])
        )
    ]
    samples = random.sample(filtered, min(n, len(filtered)))
    return [
        {
            "question":     d["issue"],
            "ground_truth": d["resolution"],
            "contexts":     [d["issue"], d["resolution"]],
            "agent":        "order_agent",
            "intent":       d["intents"][0]["intent"],
        }
        for d in samples
    ]

def load_escalation_samples(n: int = 3) -> list[dict]:
    with open(DATA_DIR / "ticket_history.json", encoding="utf-8") as f:
        docs = json.load(f)["documents"]
    escalation_intents = {"get_refund", "return_request", "seller_complaint", "payment_issue", "complaint"}
    filtered = [
        d for d in docs
        if any(
            i["intent"] in escalation_intents and i["primary"]
            for i in d.get("intents", [])
        )
    ]
    samples = random.sample(filtered, min(n, len(filtered)))
    return [
        {
            "question":     d["issue"],
            "ground_truth": d["resolution"],
            "contexts":     [d["issue"], d["resolution"]],
            "agent":        "escalation_agent",
            "intent":       d["intents"][0]["intent"],
        }
        for d in samples
    ]

def build_eval_dataset() -> list[dict]:
    dataset = (
        load_faq_samples(1) +
        load_product_samples(1) +
        load_promo_samples(1) +
        load_order_samples(1) +
        load_escalation_samples(1)
    )
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    dataset = build_eval_dataset()
    print(f"Total eval samples: {len(dataset)}")
    for i, s in enumerate(dataset, 1):
        print(f"[{i}] agent={s['agent']} | intent={s['intent']}")
        print(f"     Q: {s['question'][:80]}")