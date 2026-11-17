import atexit
import os
import time

import openai  # pyright: ignore[reportMissingImports]
from fastembed import SparseTextEmbedding, TextEmbedding  # pyright: ignore[reportMissingImports]
from openai import OpenAI  # pyright: ignore[reportMissingImports]
from qdrant_client import QdrantClient, models  # pyright: ignore[reportMissingImports]

NAME = "papers"
MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

QDRANT_URL = os.environ.get("QDRANT_URL")
if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL, api_key=os.environ["QDRANT_API_KEY"])
else:
    client = QdrantClient(path="C:/qdrant_data")  # local fallback for development
atexit.register(client.close)  # removes the shutdown traceback
dense = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse = SparseTextEmbedding("Qdrant/bm25")
llm = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.environ["LLM_API_KEY"],
    max_retries=1,
)


def to_sparse(e):
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def hybrid(query, k=5, year_from=None):
    flt = None
    if year_from:
        flt = models.Filter(must=[
            models.FieldCondition(key="year", range=models.Range(gte=year_from))
        ])
    dq = next(iter(dense.query_embed(query))).tolist()
    sq = to_sparse(next(iter(sparse.query_embed(query))))
    res = client.query_points(
        NAME,
        prefetch=[
            models.Prefetch(query=dq, using="dense", limit=20, filter=flt),
            models.Prefetch(query=sq, using="sparse", limit=20, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return [p.payload for p in res.points]


_last_call = 0.0
MIN_GAP = 2.5  # seconds between calls: stays under 30 requests/minute

FALLBACK_MODELS = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]  # tried in order if MODEL keeps failing


def ask_llm(prompt, json_mode=False, retries=3):
    global _last_call
    for model in [MODEL] + FALLBACK_MODELS:
        for attempt in range(retries):
            wait = MIN_GAP - (time.time() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.time()
            try:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                r = llm.chat.completions.create(**kwargs)  # pyright: ignore[reportCallIssue]
                if model != MODEL:
                    print(f"  (used fallback model {model})")
                return r.choices[0].message.content
            except openai.RateLimitError as e:
                if "per day" in str(e).lower():
                    break  # this model's daily quota is gone, try the next one
                print("Per-minute limit hit, waiting 20s...")
                time.sleep(20)
            except (openai.APIConnectionError, openai.InternalServerError) as e:
                pause = 2 ** attempt * 2
                print(f"LLM busy on {model} ({type(e).__name__}: {str(e)[:100]}), retrying in {pause}s...")
                time.sleep(pause)
        print(f"{model} unavailable, trying next model...")
    raise RuntimeError("All configured models are unavailable. Try again in a few minutes.")
