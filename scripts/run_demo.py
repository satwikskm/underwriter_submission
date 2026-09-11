"""
Runs the full pipeline over every bundled sample submission from the command
line -- no FastAPI server needed. Auto-approves every case at the review
gate so you can see the whole pipeline end to end in one shot.

Usage:
    python scripts/run_demo.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import compiled_graph
from app.sample_data import list_sample_submissions, load_sample_submission


def run_one(name: str):
    print(f"\n{'=' * 70}\nSUBMISSION: {name}\n{'=' * 70}")

    config = {"configurable": {"thread_id": name}}
    initial_state = load_sample_submission(name)

    compiled_graph.invoke(initial_state, config=config)
    snapshot = compiled_graph.get_state(config)
    state = snapshot.values

    print(f"Line of business : {state.get('line_of_business')}")
    print(f"Risk score        : {state.get('risk_score')}/100")
    print(f"Triage decision   : {state.get('triage_decision')}")
    print(f"Triage reasoning  : {state.get('triage_reasoning')}")

    # Auto-approve at the review gate to demonstrate the full pipeline
    compiled_graph.update_state(config, {
        "human_decision": "approve",
        "reviewer_notes": "Auto-approved by demo script.",
    })
    compiled_graph.invoke(None, config=config)
    snapshot = compiled_graph.get_state(config)
    state = snapshot.values

    print(f"\nFinal output summary:\n{state.get('output_summary')}")

    mock_stages = [e["stage"] for e in state["audit_log"] if e.get("used_mock_llm")]
    if mock_stages:
        print(f"\n(Note: mock LLM fallback was used for: {', '.join(mock_stages)} "
              f"-- install/start Ollama for real model output)")

    return state


def main():
    results = {}
    for name in list_sample_submissions():
        results[name] = run_one(name)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, state in results.items():
        print(f"{name}: {state.get('triage_decision')} (score {state.get('risk_score')})")


if __name__ == "__main__":
    main()
