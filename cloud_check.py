from core import hybrid

for d in hybrid("optimal chunk size for retrieval-augmented generation", k=3):
    print(d["year"], "|", d["title"][:70]) # pyright: ignore[reportOptionalSubscript]
print("year filter:", [d["year"] for d in hybrid("RAG frameworks", k=5, year_from=2026)]) # pyright: ignore[reportOptionalSubscript]
