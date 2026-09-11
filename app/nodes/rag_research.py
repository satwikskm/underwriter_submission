"""Knowledge agent: persistent retrieval + corrective retrieval + evidence synthesis."""
from app.rag.retriever import corrective_retrieve
from app.llm import call_llm_json

PROMPT="""You are the underwriting knowledge analyst. Summarize ONLY the retrieved guideline evidence into 3-5 concise bullets. Return ONLY JSON: {{"summary":"..."}}. Do not invent rules. If evidence is weak or absent, say so.

Retrieved evidence:
{evidence}
"""

def rag_research_node(state:dict)->dict:
    plan=state.get("agent_plan",{})
    queries=plan.get("queries",[]) or []
    results=[]; corrections=[]; seen=set()
    for q in queries[:3]:
        packet=corrective_retrieve(q,top_k=3)
        if packet.get("corrected"):
            corrections.append({"original_query":q,"corrected_query":packet.get("corrected_query"),"quality":packet.get("quality")})
        for item in packet.get("results",[]):
            key=(item["source"],item["section"])
            if key not in seen:
                seen.add(key); results.append({"query":q,**item})
    evidence="\n\n".join(f"[{x['source']} | {x['section']} | relevance={x['relevance']}]\n{x['text']}" for x in results[:8]) or "No matching guideline evidence retrieved."
    fallback={"summary":"No strong guideline evidence was retrieved; proceed using deterministic risk rules and human review."}
    summary,mock=call_llm_json(PROMPT.format(evidence=evidence[:9000]),fallback,role="reasoning",temperature=0.1)
    return {"rag_results":results,"rag_summary":summary.get("summary"),"rag_corrections":corrections,"audit_log":[{"stage":"rag_research","model_role":"reasoning","used_mock_llm":mock,"retrieved_documents":len(results),"corrective_queries":len(corrections)}]}
