from core import hybrid
from step6 import rewrite

q = "Only papers after 2025: what new RAG frameworks have been proposed?"
bad = "Greek language models musicology"
new = rewrite(q, bad, hybrid(bad))
print("rewritten:", new)
