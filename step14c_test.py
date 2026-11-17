import step10

step10.grade = lambda question, docs: (docs[:4], False)   # always: 4 relevant papers, "not sufficient"
step10.rewrite = lambda question, query, docs: query      # no rewriting

out = step10.run("What does research say about the best chunk size for RAG?")
print("\n" + out["text"][:500])
