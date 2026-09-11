"""Small helpers for LangSmith configuration and safe UI diagnostics."""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv(override=True)


def langsmith_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "false").lower() == "true" and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


def langsmith_project() -> str:
    return os.getenv("LANGSMITH_PROJECT", "underwriting-agent")
