from fastembed import SparseTextEmbedding, TextEmbedding  # pyright: ignore[reportMissingImports]
from qdrant_client import QdrantClient, models  # pyright: ignore[reportMissingImports]

client = QdrantClient(path="C:/qdrant_data")  # local mode, no Docker needed
dense = TextEmbedding("BAAI/bge-small-en-v1.5")   # 384 dims
sparse = SparseTextEmbedding("Qdrant/bm25")

NAME = "demo"
if not client.collection_exists(NAME):
    client.create_collection(
        NAME,
        vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )

docs = [
    "RAG combines a retriever with a language model to ground answers in documents.",
    "BM25 is a keyword-based ranking function used in classic search engines.",
    "Agents let an LLM decide which tool to call and when to stop.",
    "Qdrant is a vector database that supports hybrid search.",
]

def to_sparse(e):
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

points = [
    models.PointStruct(
        id=i,
        vector={"dense": d.tolist(), "sparse": to_sparse(s)},
        payload={"text": t},
    )
    for i, (t, d, s) in enumerate(zip(docs, dense.embed(docs), sparse.embed(docs)))
]
client.upsert(NAME, points=points)

# hybrid search
q = "how do search engines rank keywords?"
dq = next(iter(dense.query_embed(q))).tolist()
sq = to_sparse(next(iter(sparse.query_embed(q))))

res = client.query_points(
    NAME,
    prefetch=[
        models.Prefetch(query=dq, using="dense", limit=10),
        models.Prefetch(query=sq, using="sparse", limit=10),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=3,
)
for p in res.points:
    print(round(p.score, 3), p.payload["text"])
