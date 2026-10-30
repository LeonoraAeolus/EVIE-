"""Check your Nebius Token Factory key and find the exact Nemotron model ids.

    python scripts/check_nebius.py            # list models, try a tiny chat with NEMOTRON_MODEL
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI  # noqa: E402

from tesla_lab.config import get_settings  # noqa: E402

s = get_settings("live")
if not s.nebius_api_key:
    sys.exit("Put NEBIUS_API_KEY in .env first (https://tokenfactory.nebius.com)")
client = OpenAI(base_url=s.nebius_base_url, api_key=s.nebius_api_key)
ids = sorted(m.id for m in client.models.list().data)
nv = [i for i in ids if "nemotron" in i.lower() or i.lower().startswith("nvidia/")]
print(f"{len(ids)} models on your key. NVIDIA / Nemotron ones:")
for i in nv:
    print("  ", i)
math_models = [i for i in ids if "openmath" in i.lower()]
print("OpenMath-Nemotron on Token Factory:", math_models or "not listed")
if s.model not in ids:
    print(f"\n!! NEMOTRON_MODEL={s.model} is not in the list. Set it in .env to one of the ids above.")
    sys.exit(1)
r = client.chat.completions.create(model=s.model, messages=[{"role": "user", "content": "Reply with the word OK."}], max_tokens=200)
print(f"\nChat test with {s.model}:", (r.choices[0].message.content or "").strip()[:200])
