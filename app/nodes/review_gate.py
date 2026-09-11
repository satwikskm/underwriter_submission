"""
Stage 5: human review gate.

This node itself does very little -- its job is to be the point where the
graph pauses (see graph.py's interrupt_before=["review_gate"]). Execution
stops before this node runs; a human decision gets written into state via
graph.update_state(...), and only then does this node run and the graph
continue on to output generation.

No submission reaches "output" without a human_decision being present.
"""


def review_gate_node(state: dict) -> dict:
    decision = state.get("human_decision")
    notes = state.get("reviewer_notes", "")

    if decision is None:
        # Shouldn't normally happen -- the graph is configured to interrupt
        # before this node runs until a decision has been supplied.
        status = "pending_review"
    elif decision == "approve":
        status = "approved"
    elif decision == "escalate":
        status = "escalated"
    elif decision == "edit":
        status = "approved_with_edits"
    else:
        status = "pending_review"

    return {
        "review_status": status,
        "audit_log": [{
            "stage": "human_review_gate",
            "human_decision": decision,
            "reviewer_notes": notes,
            "resulting_status": status,
        }],
    }
