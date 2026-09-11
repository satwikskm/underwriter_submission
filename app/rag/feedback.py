"""Human feedback storage for corrective-RAG learning."""
from pathlib import Path
import json,time
ROOT=Path(__file__).resolve().parents[2]
FEEDBACK_PATH=ROOT/"data"/"rag_feedback.jsonl"

def record_feedback(query:str,source:str,section:str,label:int,notes:str=""):
    FEEDBACK_PATH.parent.mkdir(parents=True,exist_ok=True)
    row={"timestamp":time.time(),"query":query,"source":source,"section":section,"label":int(label),"notes":notes}
    with FEEDBACK_PATH.open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
    return row

def load_feedback():
    if not FEEDBACK_PATH.exists(): return []
    return [json.loads(x) for x in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
