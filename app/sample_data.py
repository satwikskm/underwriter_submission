"""Bundled synthetic submission loader."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=ROOT/"data"/"sample_submissions"

def list_sample_submissions()->list[str]: return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("submission_"))

def load_sample_submission(name:str)->dict:
    folder=DATA_DIR/name
    if not folder.exists(): raise FileNotFoundError(f"Unknown submission: {name}")
    meta=json.loads((folder/"metadata.json").read_text())
    return {"submission_id":name,"acord_text":(folder/"acord_125.txt").read_text(),"sov_csv_path":str(folder/"sov.csv"),"loss_run_csv_path":str(folder/"loss_run.csv"),"_occupancy_hint":meta["occupancy_class"],"audit_log":[{"stage":"submission_loaded","source":str(folder),"synthetic":meta.get("synthetic",False)}]}
