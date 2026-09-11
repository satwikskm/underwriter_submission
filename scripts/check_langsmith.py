"""Check that LangSmith tracing is configured in the current environment."""
import os

keys = ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]
for key in keys:
    value = os.getenv(key, "")
    if key == "LANGSMITH_API_KEY":
        value = ("configured" if value else "MISSING")
    print(f"{key}={value}")

try:
    from langsmith import traceable
    print("langsmith.traceable=OK")
except Exception as exc:
    print(f"langsmith.traceable=ERROR: {exc}")

if not os.getenv("LANGSMITH_API_KEY"):
    print("\nSet LANGSMITH_API_KEY before expecting traces in LangSmith.")
elif os.getenv("LANGSMITH_TRACING", "").lower() != "true":
    print("\nSet LANGSMITH_TRACING=true before expecting traces.")
else:
    print("\nLangSmith tracing is configured. Run a submission, then open the project in LangSmith.")
