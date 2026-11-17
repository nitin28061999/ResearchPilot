import json

from core import ask_llm, hybrid

MAX_REWRITES = 2

GRADER_PROMPT = """Question: {question}

Retrieved documents:
{docs}

Return JSON: {"relevant_ids": [<numbers of the documents that help answer the question>], "sufficient": <true if the relevant documents give enough information to write a useful answer, even if they don't cover every aspect of the question; false only if they are missing the main point>}"""

REWRITE_PROMPT = """A search query did not retrieve documents that answer the question.
Question: {question}
Previous query: {query}
Titles retrieved: {titles}
Write a better search query as plain natural-language keywords (different keywords, more specific or more general as needed). Do not use search operators, field names like all: or title:, AND/OR, or date ranges. Dates are handled separately.
Return JSON: {"query": "..."}"""


def fmt(docs, n=900):
    return "\n\n".join(f"[{i+1}] {d['title']}\n{d['abstract'][:n]}" for i, d in enumerate(docs))


def parse(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def grade(question, docs):
    prompt = GRADER_PROMPT.replace("{question}", question).replace("{docs}", fmt(docs))
    g = parse(ask_llm(prompt, json_mode=True))
    ids = [i for i in g.get("relevant_ids", []) if isinstance(i, int) and 1 <= i <= len(docs)]
    kept = [docs[i - 1] for i in ids]
    return kept, bool(g.get("sufficient")) and bool(kept)


def rewrite(question, query, docs):
    titles = "; ".join(d["title"] for d in docs)
    prompt = (REWRITE_PROMPT.replace("{question}", question)
              .replace("{query}", query).replace("{titles}", titles))
    return parse(ask_llm(prompt, json_mode=True)).get("query") or query


def agentic_retrieve(question, query, year_from=None):
    for attempt in range(MAX_REWRITES + 1):
        docs = hybrid(query, year_from=year_from)
        kept, ok = grade(question, docs)
        print(f"  attempt {attempt+1}: query='{query}' | kept {len(kept)}/{len(docs)} | sufficient={ok}")
        if ok:
            return kept, True
        if attempt < MAX_REWRITES:
            query = rewrite(question, query, docs)
            year_from = None  # drop the year filter on retry in case it was too narrow
    return kept, False


if __name__ == "__main__":
    print("TEST 1: deliberately bad query for an in-domain question")
    docs, ok = agentic_retrieve(
        "How does chunk size affect RAG performance?", query="Greek language models musicology"
    )
    print("found:", ok, [d["title"][:50] for d in docs])

    print("\nTEST 2: question the database cannot answer")
    docs, ok = agentic_retrieve(
        "How do transformers predict protein structures?",
        query="transformers protein structure prediction",
    )
    print("found:", ok, [d["title"][:50] for d in docs])
