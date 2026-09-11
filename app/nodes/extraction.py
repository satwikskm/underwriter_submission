"""
Stage 2: document extraction.

Structured files (SOV, loss run) are parsed directly with pandas -- there's
no need to spend LLM tokens turning a CSV back into JSON. The LLM is used
only where it earns its keep: pulling structured fields out of the free-text
ACORD-style form.
"""
import pandas as pd
from app.llm import call_llm_json

PROMPT = """Extract the following fields from this insurance application text.
Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"named_insured": "<string>",
  "property_address": "<string>",
  "requested_effective_date": "<string>",
  "construction_type": "<string>",
  "year_built": "<string or null>",
  "requested_tiv": <number, total insured value requested, no currency symbol>}}

Application text:
---
{text}
---
"""


def _mock_extraction(text: str) -> dict:
    """Very rough line-scanning fallback if the LLM is unavailable."""
    fields = {"named_insured": None, "property_address": None,
              "requested_effective_date": None, "construction_type": None,
              "year_built": None, "requested_tiv": None}
    for line in text.splitlines():
        low = line.lower().strip()
        # Match on field labels at the START of the line only -- this avoids
        # false hits from the same keyword appearing later in free-text
        # paragraphs (e.g. "construction" inside a narrative description).
        if low.startswith("named insured"):
            fields["named_insured"] = line.split(":", 1)[-1].strip()
        elif low.startswith("property address") or low.startswith("location address"):
            fields["property_address"] = line.split(":", 1)[-1].strip()
        elif low.startswith("effective date"):
            fields["requested_effective_date"] = line.split(":", 1)[-1].strip()
        elif low.startswith("construction type"):
            fields["construction_type"] = line.split(":", 1)[-1].strip()
        elif low.startswith("year built"):
            fields["year_built"] = line.split(":", 1)[-1].strip()
        elif low.startswith("total insured value"):
            digits = "".join(c for c in line if c.isdigit())
            fields["requested_tiv"] = int(digits) if digits else None
    return fields


def extraction_node(state: dict) -> dict:
    acord_fields, used_mock = call_llm_json(
        PROMPT.format(text=state["acord_text"][:4000]),
        mock_fallback=_mock_extraction(state["acord_text"]),
    )

    sov_df = pd.read_csv(state["sov_csv_path"])
    loss_run_df = pd.read_csv(state["loss_run_csv_path"])

    extracted_data = {
        "acord": acord_fields,
        "sov_summary": {
            "num_buildings": len(sov_df),
            "total_building_value": float(sov_df["building_value"].sum()),
            "total_contents_value": float(sov_df["contents_value"].sum()),
        },
        "loss_run_summary": {
            "num_claims": len(loss_run_df),
            "total_incurred": float(loss_run_df["incurred_amount"].sum()),
            "years_covered": int(loss_run_df["policy_year"].nunique()) if len(loss_run_df) else 0,
        },
    }

    return {
        "extracted_data": extracted_data,
        "audit_log": [{
            "stage": "document_extraction",
            "used_mock_llm": used_mock,
            "sov_rows": len(sov_df),
            "loss_run_rows": len(loss_run_df),
        }],
    }
