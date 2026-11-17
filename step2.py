# sourcery skip: list-comprehension
import arxiv  # pyright: ignore[reportMissingImports]

client = arxiv.Client(page_size=20, delay_seconds=3, num_retries=3)
search = arxiv.Search(
    query='abs:"retrieval augmented generation" AND (cat:cs.CL OR cat:cs.IR OR cat:cs.AI)',
    max_results=20,
    sort_by=arxiv.SortCriterion.Relevance,
)

papers = []
for r in client.results(search):
    papers.append({
        "id": r.get_short_id(),
        "title": " ".join(r.title.split()),
        "abstract": " ".join(r.summary.split()),
        "authors": ", ".join(a.name for a in r.authors[:3]),
        "year": r.published.year,
        "url": r.entry_id,
    })

print(f"fetched {len(papers)} papers\n")
for p in papers[:5]:
    print(p["year"], "|", p["title"])
    print("   ", p["abstract"][:150], "...\n")
