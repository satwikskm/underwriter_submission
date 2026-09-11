"""Shared LangGraph state for V3: multi-model + RAG + ML-ready underwriting."""
import operator
from typing import TypedDict, List, Optional, Annotated

class SubmissionState(TypedDict, total=False):
    submission_id: str
    acord_text: str
    sov_csv_path: str
    loss_run_csv_path: str
    _occupancy_hint: str
    line_of_business: Optional[str]
    submission_type: Optional[str]
    classification_confidence: Optional[float]
    extracted_data: Optional[dict]
    enrichment_data: Optional[dict]
    agent_plan: Optional[dict]
    rag_results: Optional[list]
    rag_summary: Optional[str]
    rag_corrections: Optional[list]
    features: Optional[dict]
    ml_prediction: Optional[dict]
    risk_score: Optional[float]
    triage_decision: Optional[str]
    triage_reasoning: Optional[str]
    decision_summary: Optional[str]
    human_decision: Optional[str]
    reviewer_notes: Optional[str]
    review_status: Optional[str]
    output_summary: Optional[str]
    audit_log: Annotated[List[dict], operator.add]
