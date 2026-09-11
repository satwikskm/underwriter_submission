"""Deterministic risk score + optional ML prediction."""
from app.llm import call_llm_json
from app.ml.predictor import predict_referral

PROMPT="""Explain the underwriting risk for a human underwriter. Return ONLY JSON: {{"triage_reasoning":"2-4 sentences"}}. Use only supplied evidence. Mention ML probability only if present.
Score={score}
Rule decision={decision}
Loss ratio={loss_ratio}
Multiplier={multiplier}
ML prediction={ml}
RAG evidence={rag}
"""

def _compute(features):
    base=min(features["loss_ratio"]*100,60)
    mult=min(max((features["combined_multiplier"]-1)*40,0),40)
    protection=-8 if features["sprinkler_present"] else 5
    score=round(max(min(base+mult+protection,100),0),1)
    if score<25: decision="auto_quote"
    elif score<55: decision="refer"
    else: decision="auto_decline_candidate"
    return score,decision

def risk_scoring_node(state:dict)->dict:
    features=state["features"]; score,decision=_compute(features)
    ml=predict_referral(features)
    loss_ratio=features["loss_ratio"]
    fallback={"triage_reasoning":f"Rule-based score {score}/100 reflects a loss ratio of {loss_ratio:.3f}, a combined risk multiplier of {features['combined_multiplier']:.2f}, and protection characteristics. Preliminary triage is {decision.replace('_',' ')}."}
    reasoning,mock=call_llm_json(PROMPT.format(score=score,decision=decision,loss_ratio=round(loss_ratio,3),multiplier=features["combined_multiplier"],ml=ml,rag=state.get("rag_summary")),fallback,role="reasoning")
    return {"risk_score":score,"triage_decision":decision,"triage_reasoning":reasoning.get("triage_reasoning"),"ml_prediction":ml,"review_status":"pending_review","audit_log":[{"stage":"risk_scoring","model_role":"reasoning","used_mock_llm":mock,"risk_score":score,"triage_decision":decision,"ml_prediction":ml}]}
