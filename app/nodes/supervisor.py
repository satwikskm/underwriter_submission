"""Agentic workflow planner: decides which knowledge/research paths are needed."""
from app.llm import call_llm_json

PROMPT="""You are the underwriting workflow supervisor. Decide what research is needed for this submission. Return ONLY JSON:
{{"queries":["query 1","query 2"],"needs_claims_review":true,"priority":"normal|high","reason":"short reason"}}

Submission facts:
LOB={lob}
Occupancy={occupancy}
Construction={construction}
TIV={tiv}
Claims={claims}
Loss ratio={loss_ratio}
"""

def supervisor_node(state:dict)->dict:
    ex=state["extracted_data"]; en=state["enrichment_data"]
    claims=ex["loss_run_summary"]["num_claims"]; tiv=ex["acord"].get("requested_tiv")
    lr=ex["loss_run_summary"]["total_incurred"]/(tiv or 1)
    occ=en["occupancy_class"]
    fallback={"queries":[f"{occ} underwriting risk guidelines",f"{state.get('line_of_business')} referral rules"],"needs_claims_review":claims>0,"priority":"high" if claims>=2 or lr>=0.25 else "normal","reason":"Claims and risk characteristics determine whether deeper guideline review is needed."}
    result,mock=call_llm_json(PROMPT.format(lob=state.get("line_of_business"),occupancy=occ,construction=ex["acord"].get("construction_type"),tiv=tiv,claims=claims,loss_ratio=round(lr,3)),fallback,role="fast")
    queries=result.get("queries") or fallback["queries"]
    return {"agent_plan":{**result,"queries":queries},"audit_log":[{"stage":"supervisor","model_role":"fast","used_mock_llm":mock,"plan":result}]}
