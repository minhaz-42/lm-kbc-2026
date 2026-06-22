"""List models available on the NVIDIA OpenAI-compatible API.
    NVIDIA_API_KEY=nvapi-... python solution/nvidia_list.py
Use it to pick a <=32B open-weight model (e.g. a Qwen / Gemma-2-27B / Mistral-Small-24B)."""
import os, sys
from openai import OpenAI

key = os.environ.get("NVIDIA_API_KEY")
if not key:
    sys.exit("set NVIDIA_API_KEY=nvapi-...")
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
ids = sorted(m.id for m in client.models.list().data)
print(f"{len(ids)} models available:\n")
for i in ids:
    print(" ", i)
