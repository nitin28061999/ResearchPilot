from core import ask_llm
from step5 import route
from step7 import retrieve_with_fallback

ANSWER_PROMPT = """Answer the question using ONLY the numbered sources below. Cite sources inline like [1], [2].
If the sources don't contain enough information, say so plainly instead of guessing.
{note}
Question: {question}

Sources:
{sources}"""


def format_sources(docs):
    parts = []
    for i, d in enumerate(docs):
        text = (d.get("abstract") or d.get("text") or "")[:1200]
        year = f" ({d['year']})" if d.get("year") else ""
        parts.append(f"[{i+1}] {d['title']}{year}\n{text}")
    return "\n\n".join(parts)


def generate(question, docs, source):
    note = (
        "These sources are web search snippets, not research papers. Say the answer comes from the web."
        if source == "web" else ""
    )
    prompt = (
        ANSWER_PROMPT.replace("{note}", note)
        .replace("{question}", question)
        .replace("{sources}", format_sources(docs))
    )
    return ask_llm(prompt)


def answer(question):
    r = route(question)
    print("route:", r)
    if r["route"] == "direct":
        return ask_llm(question), []
    docs, source = retrieve_with_fallback(question, r["query"], r["year_from"])
    if not docs:
        return "I couldn't find sources to answer that.", []
    return generate(question, docs, source), docs


if __name__ == "__main__":
    for q in [
        # "What does research say about the best chunk size for RAG?",
        "How do transformers predict protein structures?",
        "What is the capital of France?",
    ]:
        print("=" * 70)
        print("Q:", q)
        text, docs = answer(q)
        print("\n" + text)
        for i, d in enumerate(docs, 1):
            print(f"  [{i}] {d['title'][:70]}  {d.get('url', '')}")
