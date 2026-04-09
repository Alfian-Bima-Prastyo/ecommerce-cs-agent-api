from langchain_core.tools import tool


def make_rag_tool(retriever, agent_name: str):

    def _search(query: str) -> str:
        results = retriever.retrieve(query=query, agent=agent_name, top_k=5)
        if not results:
            return "Tidak ada informasi relevan di knowledge base."

        parts = []
        for i, r in enumerate(results, 1):
            p = r["payload"]
            content = (
                p.get("answer") or
                p.get("resolution") or
                p.get("description") or
                " ".join(p.get("steps", [])) or
                ""
            )
            doc_id = r.get("doc_id") or p.get("doc_id", f"doc-{i}")
            parts.append(f"[{i}] (doc_id:{doc_id})\n{content}")

        return "\n\n".join(parts)

    _search.__name__ = f"search_{agent_name.replace('_agent', '')}_kb"
    _search.__doc__  = (
        f"Cari informasi dari knowledge base {agent_name}. "
        "Gunakan tool ini PERTAMA untuk mendapat konteks sebelum tool lain."
    )
    return tool(_search)