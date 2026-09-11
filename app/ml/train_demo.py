"""Train the bundled illustrative referral model without invoking any LLM."""
from pathlib import Path
import json, re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
from app.sample_data import list_sample_submissions, load_sample_submission
from app.nodes.enrichment import _lookup_industry_factor, _lookup_construction_factor
from app.ml.predictor import FEATURE_ORDER, MODEL_PATH, METADATA_PATH

def _field(text, label):
    m=re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", text, flags=re.I|re.M)
    return m.group(1).strip() if m else ""

def build_rows():
    rows=[]
    for name in list_sample_submissions():
        s=load_sample_submission(name)
        text=s["acord_text"]
        tiv=float(re.sub(r"[^0-9.]", "", _field(text,"Total Insured Value (TIV) Requested")) or 0)
        year=float(re.sub(r"[^0-9]", "", _field(text,"Year Built")) or 0)
        construction=_field(text,"Construction Type")
        sov=pd.read_csv(s["sov_csv_path"]); loss=pd.read_csv(s["loss_run_csv_path"])
        claims=len(loss); incurred=float(loss["incurred_amount"].sum()) if len(loss) else 0.0
        tiv=tiv or float(sov["building_value"].sum()) or 1.0
        industry=_lookup_industry_factor(s["_occupancy_hint"])
        construction_factor=_lookup_construction_factor(construction)
        features={"tiv":tiv,"claims_count":claims,"total_incurred":incurred,"loss_ratio":incurred/tiv,"industry_factor":industry,"construction_factor":construction_factor,"combined_multiplier":industry*construction_factor,"year_built":year,"sprinkler_present":1.0 if "sprinkler" in text.lower() else 0.0}
        meta=json.loads((Path(s["sov_csv_path"]).parent/"metadata.json").read_text())
        target=1 if meta["expected_demo_outcome"].startswith(("refer","auto_decline")) else 0
        rows.append({**features,"target":target,"submission_id":name})
    return pd.DataFrame(rows)

def main():
    df=build_rows()
    if df["target"].nunique()<2: raise RuntimeError("Need both classes")
    X=df[FEATURE_ORDER]; y=df["target"]
    model=Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(max_iter=2000,class_weight="balanced"))])
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    scores=cross_val_score(model,X,y,cv=cv,scoring="roc_auc")
    model.fit(X,y)
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,MODEL_PATH)
    metadata={"model_version":"demo-logistic-v2","training_rows":len(df),"positive_rate":float(y.mean()),"cv_roc_auc_mean":float(scores.mean()),"cv_roc_auc_std":float(scores.std()),"features":FEATURE_ORDER,"synthetic":True,"warning":"Demonstration artifact only; replace with validated historical outcomes before production use."}
    METADATA_PATH.write_text(json.dumps(metadata,indent=2))
    print(json.dumps({"artifact":str(MODEL_PATH),**metadata},indent=2))

if __name__=="__main__": main()
