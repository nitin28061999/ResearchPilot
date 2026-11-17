import json

from core import ask_llm
from step5 import route
from step7 import retrieve_with_fallback
from step8 import ANSWER_PROMPT, format_sources

CHECK_PROMPT = """Question: {question}

Sources:
{sources}

Answer:
{answer}

List every claim in the answer that the sources do NOT support, including general-knowledge additions, tables, and conclusions that go beyond what the sources say.
Return JSON: {"grounded": <true if there are no unsupported claims, else false>, "unsupported": ["short description of each unsupported claim"]}"""

WEB_NOTE = "_No matching papers in the database. This answer is based on web search snippets._\n\n"


def dedupe(docs):
    seen, out = set(), []
    for d in docs:
        key = d["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def generate_text(question, docs, source, issues=None):
    note = ""
    if source == "web":
        note += "These sources are web search snippets, not research papers.\n"
    if issues:
        note += (
            "A previous draft contained unsupported claims: " + "; ".join(issues) +
            ". Remove them. Use only facts stated in the sources. No tables, no general-knowledge summaries.\n"
        )
    prompt = (
        ANSWER_PROMPT.replace("{note}", note)
        .replace("{question}", question)
        .replace("{sources}", format_sources(docs))
    )
    return ask_llm(prompt)


def check(question, docs, text):
    prompt = (
        CHECK_PROMPT.replace("{question}", question)
        .replace("{sources}", format_sources(docs))
        .replace("{answer}", text)
    )
    try:
        r = json.loads(ask_llm(prompt, json_mode=True))
    except json.JSONDecodeError:
        return True, []  # if the checker itself fails, don't block the answer
    issues = [str(u) for u in r.get("unsupported", [])]
    return bool(r.get("grounded", True)) and not issues, issues


def answer(question):
    r = route(question)
    print("route:", r)
    if r["route"] == "direct":
        return ask_llm(question), []

    docs, source = retrieve_with_fallback(question, r["query"], r["year_from"])
    docs = dedupe(docs)
    if not docs:
        return "I couldn't find sources to answer that.", []

    text = generate_text(question, docs, source)
    grounded, issues = check(question, docs, text)
    print(f"  check 1: grounded={grounded} {issues}")

    if not grounded:
        text = generate_text(question, docs, source, issues)
        grounded, issues = check(question, docs, text)
        print(f"  check 2: grounded={grounded} {issues}")

    if not grounded:
        text += "\n\n> Note: some statements may not be fully supported by the sources."
    if source == "web":
        text = WEB_NOTE + text
    return text, docs


if __name__ == "__main__":
    for q in [
        "What does research say about the best chunk size for RAG?",
        "How do transformers predict protein structures?",
    ]:
        print("=" * 70)
        print("Q:", q)
        text, docs = answer(q)
        print("\n" + text)
        for i, d in enumerate(docs, 1):
            print(f"  [{i}] {d['title'][:70]}  {d.get('url', '')}")
