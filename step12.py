import json
import os
import time

from core import hybrid, llm
from step8 import format_sources
from step9 import generate_text
from step10 import run

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen/qwen3.8-27b")
LIMIT = 10

QUESTIONS = [
    "What does research say about the best chunk size for RAG?",
    "How is graph-based RAG different from standard RAG?",
    "How is RAG used for financial question answering?",
    "What approaches exist for multimodal retrieval-augmented generation?",
    "What challenges arise when building RAG systems for technical documents?",
    "How can RAG help with navigating academic literature?",
    "Only papers after 2025: what new RAG frameworks have been proposed?",
    "How does RAG handle long-document question answering?",
    "How do transformers predict protein structures?",
    "What are the main ways to optimize RAG system performance?",
]

JUDGE_PROMPT = """You are a strict fact-checker.

Question: {question}

Sources:
{sources}

Answer:
{answer}

Split the answer into individual factual claims. A claim is unsupported if the sources do not state it, even if it is true in general.
Return JSON: {"total_claims": <int>, "unsupported": <int>, "examples": ["up to 3 short descriptions of unsupported claims"], "answers_question": <true if the answer substantively addresses the question, false if it mainly says the sources lack the information>}"""


def judge(question, docs, text):
    prompt = (JUDGE_PROMPT.replace("{question}", question)
              .replace("{sources}", format_sources(docs)).replace("{answer}", text))
    for _ in range(3):
        time.sleep(3)
        try:
            r = llm.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            d = json.loads(r.choices[0].message.content) # pyright: ignore[reportArgumentType]
            return {"total": int(d["total_claims"]), "unsupported": int(d["unsupported"]),
                    "examples": d.get("examples", []),
                    "answered": bool(d.get("answers_question", True))}
        except (TimeoutError, ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as e:
            print("  judge retry:", type(e).__name__)
            time.sleep(10)
    return None


def plain_rag(question):
    docs = hybrid(question, k=5)
    return generate_text(question, docs, "papers"), docs


def summarize(rows):
    for name in ("plain", "agent"):
        scored = [r[name]["judge"] for r in rows if r[name]["judge"]]
        total = sum([j["total"] for j in scored])
        bad = sum([j["unsupported"] for j in scored])
# sourcery skip: comprehension-to-generator, simplify-constant-sum
        clean = sum(1 for j in scored if j["unsupported"] == 0)
# sourcery skip: simplify-constant-sum
        answered = sum(1 for j in scored if j["answered"])
        print(f"{name:6} | answered the question: {answer_fmt(answered, len(scored))} | "
              f"fully grounded: {clean}/{len(scored)} | "
              f"unsupported claims: {bad}/{total} ({100 * bad / max(total, 1):.0f}%)")


def answer_fmt(a, n):
    return f"{a}/{n}"


if __name__ == "__main__":
    rows = []
    for q in QUESTIONS[:LIMIT]:
        print("=" * 70)
        print("Q:", q)
        ptext, pdocs = plain_rag(q)
        out = run(q)
        atext, adocs = out["text"], out.get("docs", [])
        row = {
            "question": q,
            "plain": {"text": ptext, "judge": judge(q, pdocs, ptext)},
            "agent": {"text": atext, "route": out.get("route"), "source": out.get("source"),
                      "judge": judge(q, adocs, atext) if adocs else None},
        }
        rows.append(row)
        print("  plain:", row["plain"]["judge"])
        print("  agent:", row["agent"]["judge"])
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    summarize(rows)
