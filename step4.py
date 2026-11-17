import os
import time

from fastembed import SparseTextEmbedding, TextEmbedding  # pyright: ignore[reportMissingImports]
from google import genai  # pyright: ignore[reportAttributeAccessIssue]
from google.genai import errors  # pyright: ignore[reportMissingImports]
from qdrant_client import QdrantClient, models  # pyright: ignore[reportMissingImports]

NAME = "papers"
client = QdrantClient(path="C:/qdrant_data")
dense = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse = SparseTextEmbedding("Qdrant/bm25")
llm = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.5-flash"  # change if your key has a different model

def to_sparse(e):
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

def hybrid(query, k=5):
    dq = next(iter(dense.query_embed(query))).tolist()
    sq = to_sparse(next(iter(sparse.query_embed(query))))
    res = client.query_points(
        NAME,
        prefetch=[
            models.Prefetch(query=dq, using="dense", limit=20),
            models.Prefetch(query=sq, using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return [p.payload for p in res.points]

def ask_llm(prompt, retries=4):
    for attempt in range(retries):
        try:
            return llm.models.generate_content(model=MODEL, contents=prompt).text
        except errors.ServerError:
            wait = 2 ** attempt * 2
            print(f"Gemini busy, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Gemini still unavailable. Try again later or change MODEL.")

def plain_rag(question):
    docs = hybrid(question)
    context = "\n\n".join(
        f"[{i+1}] {d['title']} ({d['year']})\n{d['abstract']}" for i, d in enumerate(docs)
    )
    prompt = (
        "Answer the question using ONLY the numbered sources below. "
        "Cite sources inline like [1]. If the sources don't contain the answer, say so.\n\n"
        f"Question: {question}\n\nSources:\n{context}"
    )
    answer = ask_llm(prompt)
    return answer, docs

for q in [
    "What does research say about the best chunk size for RAG?",
    "What is the capital of France?",
]:
    answer, docs = plain_rag(q)
    print("Q:", q)
    print(answer)
    print("sources:", [d["title"][:50] for d in docs])
    print("-" * 60)

client.close()
