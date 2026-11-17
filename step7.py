import time

from ddgs import DDGS  # pyright: ignore[reportMissingImports]
from ddgs.exceptions import DDGSException  # pyright: ignore[reportMissingImports]

from step6 import agentic_retrieve


def web_search(query, n=5):
    # try the full query, then a shortened version
    for q in [query, " ".join(query.split()[:8])]:
        for attempt in range(2):
            try:
                with DDGS() as d:
                    hits = list(d.text(q, max_results=n))
                if hits:
                    return [
                        {"title": h.get("title", ""), "url": h.get("href", ""), "text": h.get("body", "")}
                        for h in hits
                    ]
            except DDGSException as e:
                print(f"  web search error ({type(e).__name__}): {e}")
                if attempt < 1:
                    print("  retrying...")
                    time.sleep(2)
    return []


def retrieve_with_fallback(question, query, year_from=None):
    docs, ok = agentic_retrieve(question, query, year_from)
    if ok:
        return docs, "papers"
    print("  papers failed -> falling back to web search")
    try:
        return web_search(question), "web"
    except DDGSException as e:
        print("  web search failed:", e)
        return [], "none"


if __name__ == "__main__":
    q = "How do transformers predict protein structures?"
    docs, source = retrieve_with_fallback(q, query="transformers protein structure prediction")
    print("\nsource:", source)
    for d in docs:
        print("-", d["title"][:80]) # pyright: ignore[reportOptionalSubscript]
        print("  ", d["url"]) # pyright: ignore[reportOptionalSubscript]
