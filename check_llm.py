from core import MODEL, ask_llm

print("model:", MODEL)
print(ask_llm("Reply with the single word: ok"))
print(ask_llm('Return JSON: {"status": "ok"}', json_mode=True))
