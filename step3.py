import arxiv  # pyright: ignore[reportMissingImports]
from fastembed import SparseTextEmbedding, TextEmbedding  # pyright: ignore[reportMissingImports]
from qdrant_client import QdrantClient, models  # pyright: ignore[reportMissingImports]

NAME = "papers"
client = QdrantClient(path="C:/qdrant_data")
dense = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse = SparseTextEmbedding("Qdrant/bm25")

def to_sparse(e):
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

# --- 1. fetch (same as step 2, but 100 papers) ---
ax = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
search = arxiv.Search(
    query='abs:"retrieval augmented generation" AND (cat:cs.CL OR cat:cs.IR OR cat:cs.AI)',
    max_results=100,
    sort_by=arxiv.SortCriterion.Relevance,
)
papers = [
    {
        "id": r.get_short_id(),
        "title": " ".join(r.title.split()),
        "abstract": " ".join(r.summary.split()),
        "year": r.published.year,
        "url": r.entry_id,
    }
    for r in ax.results(search)
]
print("fetched", len(papers))

# --- 2. store ---
if client.collection_exists(NAME):
    client.delete_collection(NAME)  # start fresh each run while we experiment
client.create_collection(
    NAME,
    vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
    sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)

texts = [f"{p['title']}. {p['abstract']}" for p in papers]
points = [
    models.PointStruct(
        id=i,
        vector={"dense": d.tolist(), "sparse": to_sparse(s)},
        payload=p,
    )
    for i, (p, d, s) in enumerate(zip(papers, dense.embed(texts), sparse.embed(texts)))
]
client.upsert(NAME, points=points)
print("stored", client.count(NAME).count)

# --- 3. search, with and without a filter ---
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
    return res.points

q = "how should documents be split into chunks?"
print("\n--- no filter ---")
for p in hybrid(q):
    print(p.payload["year"], "|", p.payload["title"])

print("\n--- year >= 2025 ---")
for p in hybrid(q, year_from=2025):
    print(p.payload["year"], "|", p.payload["title"])

client.close()
