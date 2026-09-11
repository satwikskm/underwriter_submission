#!/usr/bin/env python3
"""Check Ollama availability and V3 model configuration."""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if load_dotenv:
    load_dotenv(os.path.join(ROOT, ".env"))

models = {
    "fast": os.getenv("OLLAMA_MODEL_FAST", "llama3.2:latest"),
    "reasoning": os.getenv("OLLAMA_MODEL_REASONING", "llama3:latest"),
    "synthesis": os.getenv("OLLAMA_MODEL_SYNTHESIS", "llama3:latest"),
}
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

print("V3 Ollama model configuration")
for role, model in models.items():
    print(f"  {role:<10} -> {model}")

print(f"\nOllama endpoint: {base_url}")
try:
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
        import json
        payload = json.load(response)
    installed = [m.get("name") for m in payload.get("models", [])]
    print("Ollama: CONNECTED")
    print("Installed models:")
    for name in installed:
        print(f"  - {name}")
    print()
    for role, model in models.items():
        if model in installed:
            print(f"OK  {role}: {model}")
        else:
            # Ollama can sometimes return a model without the explicit :latest tag.
            alt = model.split(":", 1)[0]
            matches = [x for x in installed if x.split(":", 1)[0] == alt]
            if matches:
                print(f"WARN {role}: configured {model}, installed equivalent: {matches[0]}")
            else:
                print(f"MISS {role}: {model} is not installed")
except (urllib.error.URLError, TimeoutError, OSError) as exc:
    print("Ollama: NOT REACHABLE")
    print(f"  {exc}")
    print("\nStart Ollama and run this script again.")
    sys.exit(1)
