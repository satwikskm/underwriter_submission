"""
Stage 3: enrichment.

In a real deployment this would call out to external data sources
(industry classification lookups, catastrophe zone data, prior-carrier
records). For this local demo, that's replaced with small local lookup
tables so the pipeline has no external dependency -- swap the body of
`_lookup_industry_factor` / `_lookup_cat_zone` for real API calls later
without touching any other node.
"""

INDUSTRY_RISK_FACTORS = {
    "retail_store": 1.0,
    "restaurant": 1.6,
    "warehouse_general": 1.3,
    "warehouse_hazardous_storage": 2.2,
    "office": 0.8,
}

CONSTRUCTION_RISK_FACTORS = {
    "fire_resistive": 0.7,
    "masonry_noncombustible": 0.9,
    "joisted_masonry": 1.1,
    "frame": 1.5,
}


def _lookup_industry_factor(occupancy: str) -> float:
    return INDUSTRY_RISK_FACTORS.get(occupancy, 1.2)


def _lookup_construction_factor(construction_type: str) -> float:
    key = (construction_type or "").strip().lower().replace(" ", "_").replace("-", "_")
    for known_key, factor in CONSTRUCTION_RISK_FACTORS.items():
        if known_key in key:
            return factor
    return 1.2


def enrichment_node(state: dict) -> dict:
    extracted = state["extracted_data"]
    occupancy = state.get("_occupancy_hint", "retail_store")  # set by sample data loader
    construction_type = extracted["acord"].get("construction_type", "")

    industry_factor = _lookup_industry_factor(occupancy)
    construction_factor = _lookup_construction_factor(construction_type)

    enrichment_data = {
        "occupancy_class": occupancy,
        "industry_risk_factor": industry_factor,
        "construction_risk_factor": construction_factor,
        "combined_risk_multiplier": round(industry_factor * construction_factor, 2),
    }

    return {
        "enrichment_data": enrichment_data,
        "audit_log": [{
            "stage": "enrichment",
            "used_mock_llm": False,  # deterministic lookups, no LLM involved
            "result": enrichment_data,
        }],
    }
