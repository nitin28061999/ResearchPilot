import os

from qdrant_client import QdrantClient, models

NAME = "papers"
src = QdrantClient(path="C:/qdrant_data")
dst = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])

if dst.collection_exists(NAME):
    dst.delete_collection(NAME)
dst.create_collection(
    NAME,
    vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
    sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
dst.create_payload_index(NAME, "year", models.PayloadSchemaType.INTEGER)

points, _ = src.scroll(NAME, limit=1000, with_payload=True, with_vectors=True)
dst.upsert(
    NAME,
    points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points], # pyright: ignore[reportArgumentType]
)
print("copied:", len(points), "| in cloud:", dst.count(NAME).count)
src.close()
