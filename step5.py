import json

from core import ask_llm, hybrid

ROUTER_PROMPT = """You route questions for a research assistant that has a database of arXiv papers about retrieval-augmented generation (RAG).

Routes:
- papers: the question needs information from research papers (methods, findings, comparisons, trends)
- direct: general knowledge, chit-chat, or anything unrelated to the papers

Return JSON: {"route": "papers" or "direct", "query": "<standalone search query for papers>", "year_from": <integer year or null>}
Set year_from only if the user restricts results to recent papers or to after a certain year.
"After 2024" means year_from 2025; "since 2024" or "from 2024" means 2024.

Question: {question}"""


def route(question):
    raw = ask_llm(ROUTER_PROMPT.replace("{question}", question), json_mode=True)
    try:
        r = json.loads(raw)
    except json.JSONDecodeError:
        r = {}
    if r.get("route") not in ("papers", "direct"):
        r["route"] = "papers"  # safe default
    r["query"] = r.get("query") or question
    if not isinstance(r.get("year_from"), int):
        r["year_from"] = None
    return r


def answer(question):
    r = route(question)
    print("route:", r)
    if r["route"] == "direct":
        return ask_llm(question)
    docs = hybrid(r["query"], year_from=r["year_from"])
    context = "\n\n".join(
        f"[{i+1}] {d['title']} ({d['year']})\n{d['abstract']}" for i, d in enumerate(docs)
    )
    prompt = (
        "Answer the question using ONLY the numbered sources below. "
        "Cite sources inline like [1]. If the sources don't contain the answer, say so.\n\n"
        f"Question: {question}\n\nSources:\n{context}"
    )
    return ask_llm(prompt)


if __name__ == "__main__":
    for q in [
        "What is the capital of France?",
        "What does research say about the best chunk size for RAG?",
        "Only papers after 2024: how is graph-based RAG different from normal RAG?",
        "hi, what can you help me with?",
    ]:
        print("Q:", q)
        print(answer(q)[:400])
        print("-" * 60)
