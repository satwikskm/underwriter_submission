"""Batch-evaluate all synthetic submissions without pausing for human review."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from app.graph import compiled_graph
from app.sample_data import list_sample_submissions,load_sample_submission

def main():
    rows=[]
    for name in list_sample_submissions():
        c={"configurable":{"thread_id":"batch-"+name}}; compiled_graph.invoke(load_sample_submission(name),config=c); s=compiled_graph.get_state(c).values
        # resume as demo approve
        compiled_graph.update_state(c,{"human_decision":"approve","reviewer_notes":"Batch evaluation demo."}); compiled_graph.invoke(None,config=c); s=compiled_graph.get_state(c).values
        meta=json.loads((Path(s["sov_csv_path"]).parent/"metadata.json").read_text()); expected=meta["expected_demo_outcome"].split()[0]
        predicted=s.get("triage_decision","unknown").replace("auto_quote","auto_quote").replace("auto_decline_candidate","auto_decline_candidate")
        rows.append({"submission_id":name,"expected":expected,"predicted":predicted,"score":s.get("risk_score"),"ml":s.get("ml_prediction")})
    out=Path("data/batch_results.json"); out.write_text(json.dumps(rows,indent=2)); print(f"Wrote {out} for {len(rows)} submissions.")
    print("This is workflow evaluation on synthetic labels, not model validation.")
if __name__=="__main__": main()
