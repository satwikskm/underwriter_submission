"""Create stable numeric features shared by the risk engine and future ML model."""

def feature_engineering_node(state:dict)->dict:
    ex=state["extracted_data"]; en=state["enrichment_data"]
    acord=ex["acord"]; tiv=float(acord.get("requested_tiv") or ex["sov_summary"]["total_building_value"] or 1)
    incurred=float(ex["loss_run_summary"]["total_incurred"]); claims=int(ex["loss_run_summary"]["num_claims"])
    features={
        "tiv":tiv,
        "claims_count":claims,
        "total_incurred":incurred,
        "loss_ratio":incurred/tiv if tiv else 0.0,
        "industry_factor":float(en["industry_risk_factor"]),
        "construction_factor":float(en["construction_risk_factor"]),
        "combined_multiplier":float(en["combined_risk_multiplier"]),
        "year_built":float(acord.get("year_built") or 0),
        "sprinkler_present":1.0 if "sprinkler" in state.get("acord_text","").lower() else 0.0,
    }
    return {"features":features,"audit_log":[{"stage":"feature_engineering","feature_count":len(features)}]}
