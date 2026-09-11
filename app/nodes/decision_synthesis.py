"""Final AI synthesis before human review; it does not override deterministic rules."""
from app.llm import call_llm_json

PROMPT="""You are the senior underwriting decision assistant. Synthesize the deterministic score, ML signal, and retrieved guideline evidence. Return ONLY JSON: {{"decision_summary":"4-6 sentences","key_evidence":["..."]}}. Never claim the AI made the final underwriting decision; state that human review is required.
Rule score={score}; rule triage={triage}
ML={ml}
RAG={rag}
Features={features}
"""

def decision_synthesis_node(state:dict)->dict:
    fallback={"decision_summary":f"The deterministic risk engine produced a score of {state.get('risk_score')}/100 with preliminary triage {state.get('triage_decision')}. Retrieved underwriting guidance and the available historical-loss signals should be reviewed with the submission. The system is recommending the preliminary path only and requires an authorized human underwriter for the final decision.","key_evidence":[]}
    result,mock=call_llm_json(PROMPT.format(score=state.get("risk_score"),triage=state.get("triage_decision"),ml=state.get("ml_prediction"),rag=state.get("rag_summary"),features=state.get("features")),fallback,role="synthesis")
    return {"decision_summary":result.get("decision_summary"),"audit_log":[{"stage":"decision_synthesis","model_role":"synthesis","used_mock_llm":mock,"key_evidence":result.get("key_evidence",[])}]}
