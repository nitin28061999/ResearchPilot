import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingImports]

from core import ask_llm, hybrid
from step5 import route
from step6 import MAX_REWRITES, grade, rewrite
from step7 import web_search
from step9 import WEB_NOTE, check, dedupe, generate_text


class State(TypedDict, total=False):
    question: str
    query: str
    year_from: int | None
    route: str
    docs: list
    best_docs: list
    source: str
    relevant: bool
    attempts: int
    gen_attempts: int
    issues: list
    text: str
    grounded: bool
    trace: Annotated[list, operator.add]


def route_node(s):
    r = route(s["question"])
    return {
        "route": r["route"], "query": r["query"], "year_from": r["year_from"],
        "attempts": 0, "gen_attempts": 0, "issues": [], "docs": [],
        "best_docs": [],
        "trace": [f"router -> {r['route']} | query='{r['query']}' | year_from={r['year_from']}"],
    }


def direct_node(s):
    return {"text": ask_llm(s["question"]), "docs": [], "trace": ["direct answer, no retrieval"]}


def retrieve_node(s):
    docs = hybrid(s["query"], year_from=s["year_from"])
    return {"docs": docs, "source": "papers", "trace": [f"retrieve -> {len(docs)} papers"]}


def grade_node(s):
    kept, ok = grade(s["question"], s["docs"])
    n = s["attempts"] + 1
    best = kept if len(kept) > len(s.get("best_docs") or []) else s.get("best_docs", [])
    return {
        "docs": kept if ok else s["docs"], "relevant": ok, "attempts": n, "best_docs": best,
        "trace": [f"grade -> kept {len(kept)}/{len(s['docs'])}, sufficient={ok} (attempt {n})"],
    }


def partial_node(s):
    return {"docs": s["best_docs"], "source": "papers",
            "trace": [f"partial -> using {len(s['best_docs'])} relevant papers from the best attempt"]}


def rewrite_node(s):
    q = rewrite(s["question"], s["query"], s["docs"])
    return {"query": q, "year_from": None, "trace": [f"rewrite -> '{q}'"]}


def web_node(s):
    try:
        docs = dedupe(web_search(s["question"]))
    except (ConnectionError, OSError, TimeoutError, ValueError) as e:
        docs = []
        print("web search failed:", e)
    return {"docs": docs, "source": "web", "trace": [f"web search -> {len(docs)} results"]}


def generate_node(s):
    n = s["gen_attempts"] + 1
    if not s["docs"]:
        return {"text": "I couldn't find sources to answer that.", "gen_attempts": n,
                "trace": ["generate -> no sources"]}
    text = generate_text(s["question"], s["docs"], s["source"], s["issues"] or None)
    return {"text": text, "gen_attempts": n, "trace": [f"generate -> draft {n}"]}


def check_node(s):
    if not s["docs"]:
        return {"grounded": True, "trace": ["check -> skipped"]}
    ok, issues = check(s["question"], s["docs"], s["text"])
    return {"grounded": ok, "issues": issues, "trace": [f"check -> grounded={ok}, {len(issues)} issue(s)"]}


def finalize_node(s):
    text = s["text"]
    if not s.get("grounded", True):
        text += "\n\n> Note: some statements may not be fully supported by the sources."
    if s.get("source") == "web" and s["docs"]:
        text = WEB_NOTE + text
    return {"text": text, "trace": ["finalize"]}


def after_route(s):
    return "direct" if s["route"] == "direct" else "retrieve"


def after_grade(s):
    if s["relevant"]:
        return "generate"
    if s["attempts"] <= MAX_REWRITES:
        return "rewrite"
    return "partial" if len(s["best_docs"]) >= 3 else "web"


def after_check(s):
    return "finalize" if s["grounded"] or s["gen_attempts"] >= 2 else "generate"


g = StateGraph(State)
for name, fn in [
    ("route", route_node), ("direct", direct_node), ("retrieve", retrieve_node),
    ("grade", grade_node), ("rewrite", rewrite_node), ("partial", partial_node), ("web", web_node),
    ("generate", generate_node), ("check", check_node), ("finalize", finalize_node),
]:
    g.add_node(name, fn)
g.add_edge(START, "route")
g.add_conditional_edges("route", after_route, ["direct", "retrieve"])
g.add_edge("direct", END)
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "web", "partial"])
g.add_edge("rewrite", "retrieve")
g.add_edge("partial", "generate")
g.add_edge("web", "generate")
g.add_edge("generate", "check")
g.add_conditional_edges("check", after_check, ["generate", "finalize"])
g.add_edge("finalize", END)
app = g.compile()


def run(question):
    state = {}
    for update in app.stream({"question": question}, stream_mode="updates"):
        for delta in update.values():
            for line in delta.get("trace", []):
                print("  ", line)
            state.update({k: v for k, v in delta.items() if k != "trace"})
    return state


if __name__ == "__main__":
    for q in [
        "What is the capital of France?",
        "What does research say about the best chunk size for RAG?",
    ]:
        print("=" * 70)
        print("Q:", q)
        out = run(q)
        print("\n" + out["text"])
        for i, d in enumerate(out.get("docs", []), 1):
            print(f"  [{i}] {d['title'][:70]}  {d.get('url', '')}")
