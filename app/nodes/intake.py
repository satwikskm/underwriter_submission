"""Stage 1: intake & classification."""
from app.llm import call_llm_json

PROMPT = """You are an insurance submission triage assistant.
Read the submission text below and classify it.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"line_of_business": "<commercial_property|commercial_general_liability|commercial_auto|other>",
  "submission_type": "<new_business|renewal|unknown>",
  "classification_confidence": <float between 0 and 1>}}

Submission text:
---
{text}
---
"""


def _keyword_mock(text: str) -> dict:
    """Deterministic fallback: simple keyword rules, used only if the LLM is unavailable."""
    lower = text.lower()
    if "sprinkler" in lower or "building" in lower or "property" in lower:
        lob = "commercial_property"
    elif "auto" in lower or "vehicle" in lower or "fleet" in lower:
        lob = "commercial_auto"
    elif "liability" in lower:
        lob = "commercial_general_liability"
    else:
        lob = "other"
    sub_type = "renewal" if "renewal" in lower else "new_business"
    return {"line_of_business": lob, "submission_type": sub_type, "classification_confidence": 0.6}


def intake_node(state: dict) -> dict:
    text = state["acord_text"]
    result, used_mock = call_llm_json(
        PROMPT.format(text=text[:4000]),
        mock_fallback=_keyword_mock(text),
    )

    return {
        "line_of_business": result.get("line_of_business", "other"),
        "submission_type": result.get("submission_type", "unknown"),
        "classification_confidence": result.get("classification_confidence", 0.5),
        "audit_log": [{
            "stage": "intake_classification",
            "used_mock_llm": used_mock,
            "result": result,
        }],
    }
