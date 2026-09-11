"""Stage 10: final underwriter-facing output."""
from app.llm import call_llm_json
PROMPT="""Write a concise final underwriter-facing summary. Return ONLY JSON: {{"output_summary":"4-6 sentences"}}. Include insured, TIV, risk score, preliminary triage, human decision/status, key evidence, and reviewer notes.
Insured={insured}
TIV={tiv}
Score={score}
Triage={triage}
AI synthesis={synthesis}
Review={review}
Notes={notes}
"""
def output_node(state:dict)->dict:
    acord=state["extracted_data"]["acord"]
    fallback={"output_summary":f"{acord.get('named_insured')} requested ${acord.get('requested_tiv')} of coverage. The deterministic risk score is {state.get('risk_score')}/100 with preliminary triage {state.get('triage_decision')}. {state.get('decision_summary') or state.get('triage_reasoning')} Human review status: {state.get('review_status')}. Reviewer notes: {state.get('reviewer_notes') or 'none'}."}
    result,mock=call_llm_json(PROMPT.format(insured=acord.get("named_insured"),tiv=acord.get("requested_tiv"),score=state.get("risk_score"),triage=state.get("triage_decision"),synthesis=state.get("decision_summary"),review=state.get("review_status"),notes=state.get("reviewer_notes") or "none"),fallback,role="synthesis")
    return {"output_summary":result.get("output_summary"),"audit_log":[{"stage":"output_generation","model_role":"synthesis","used_mock_llm":mock}]}
