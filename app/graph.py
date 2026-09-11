"""V3 LangGraph orchestration: multi-model agents + local RAG + ML-ready scoring."""
from __future__ import annotations
import sqlite3,time
from pathlib import Path
from typing import Callable
from langgraph.config import get_stream_writer
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn): return fn
        return decorator
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from app.state import SubmissionState
from app.nodes.intake import intake_node
from app.nodes.extraction import extraction_node
from app.nodes.enrichment import enrichment_node
from app.nodes.supervisor import supervisor_node
from app.nodes.rag_research import rag_research_node
from app.nodes.feature_engineering import feature_engineering_node
from app.nodes.risk_scoring import risk_scoring_node
from app.nodes.decision_synthesis import decision_synthesis_node
from app.nodes.review_gate import review_gate_node
from app.nodes.output import output_node
DB_PATH=Path(__file__).resolve().parent.parent/"data"/"checkpoints.sqlite"
NODE_FUNCTIONS={"intake":intake_node,"extraction":extraction_node,"enrichment":enrichment_node,"supervisor":supervisor_node,"rag_research":rag_research_node,"feature_engineering":feature_engineering_node,"risk_scoring":risk_scoring_node,"decision_synthesis":decision_synthesis_node,"review_gate":review_gate_node,"output":output_node}

def _instrument_node(name:str,fn:Callable[[dict],dict]):
    def wrapped(state:dict)->dict:
        try: writer=get_stream_writer()
        except RuntimeError: writer=lambda _:None
        start=time.perf_counter(); writer({"event":"node_started","node":name,"timestamp":time.time()})
        try:
            result=fn(state); ms=round((time.perf_counter()-start)*1000,1)
            writer({"event":"node_completed","node":name,"duration_ms":ms,"timestamp":time.time()}); return result
        except Exception as exc:
            ms=round((time.perf_counter()-start)*1000,1); writer({"event":"node_failed","node":name,"duration_ms":ms,"error":str(exc),"timestamp":time.time()}); raise
    traced=traceable(wrapped,name=f"underwriting.{name}",run_type="chain"); traced.__name__=f"instrumented_{name}"; return traced

def build_graph():
    g=StateGraph(SubmissionState)
    for n,fn in NODE_FUNCTIONS.items(): g.add_node(n,_instrument_node(n,fn))
    g.set_entry_point("intake")
    for a,b in [("intake","extraction"),("extraction","enrichment"),("enrichment","supervisor"),("supervisor","rag_research"),("rag_research","feature_engineering"),("feature_engineering","risk_scoring"),("risk_scoring","decision_synthesis"),("decision_synthesis","review_gate"),("review_gate","output")]: g.add_edge(a,b)
    g.add_edge("output",END)
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(str(DB_PATH),check_same_thread=False)
    return g.compile(checkpointer=SqliteSaver(conn),interrupt_before=["review_gate"])
compiled_graph=build_graph()
