"""
FastAPI app exposing the underwriting pipeline.

Typical flow:
  POST /submissions/{name}/run       -> runs intake through risk_scoring, pauses at review
  POST /submissions/{name}/review    -> supplies the human decision, resumes to output
  GET  /submissions/{name}           -> current state + full audit trail
  GET  /submissions                  -> list bundled sample submissions

Run with: uvicorn app.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph import compiled_graph
from app.sample_data import list_sample_submissions, load_sample_submission

app = FastAPI(title="Insurance Underwriting Submission Agent (demo)")


def _config(submission_id: str) -> dict:
    return {"configurable": {"thread_id": submission_id}}


@app.get("/submissions")
def list_submissions():
    return {"sample_submissions": list_sample_submissions()}


@app.post("/submissions/{name}/run")
def run_submission(name: str):
    try:
        initial_state = load_sample_submission(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    config = _config(name)
    compiled_graph.invoke(initial_state, config=config)
    snapshot = compiled_graph.get_state(config)
    return {"submission_id": name, "state": snapshot.values, "next": snapshot.next}


class ReviewDecision(BaseModel):
    decision: str  # approve | edit | escalate
    notes: str = ""


@app.post("/submissions/{name}/review")
def review_submission(name: str, review: ReviewDecision):
    config = _config(name)
    snapshot = compiled_graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="No such submission run yet -- call /run first")
    if "review_gate" not in snapshot.next:
        raise HTTPException(status_code=400, detail=f"Submission is not waiting for review (next={snapshot.next})")

    compiled_graph.update_state(config, {
        "human_decision": review.decision,
        "reviewer_notes": review.notes,
    })
    compiled_graph.invoke(None, config=config)  # resume from the interrupt
    snapshot = compiled_graph.get_state(config)
    return {"submission_id": name, "state": snapshot.values, "next": snapshot.next}


@app.get("/submissions/{name}")
def get_submission(name: str):
    config = _config(name)
    snapshot = compiled_graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="No such submission run yet -- call /run first")
    return {"submission_id": name, "state": snapshot.values, "next": snapshot.next}
