import os

from openai import OpenAI  # pyright: ignore[reportMissingImports]

c = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["LLM_API_KEY"])
for m in sorted(c.models.list().data, key=lambda m: m.id):
    print(m.id)
