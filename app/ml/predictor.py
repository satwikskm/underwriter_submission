"""Referral ML inference with artifact metadata and safe status reporting."""
from pathlib import Path
import json, os
MODEL_PATH=Path(__file__).resolve().parent/"artifacts"/"referral_model.joblib"
METADATA_PATH=Path(__file__).resolve().parent/"artifacts"/"referral_model_metadata.json"
FEATURE_ORDER=["tiv","claims_count","total_incurred","loss_ratio","industry_factor","construction_factor","combined_multiplier","year_built","sprinkler_present"]

def predict_referral(features:dict)->dict:
    enabled=os.getenv("ENABLE_ML_MODEL","true").lower()=="true"
    if not enabled:
        return {"enabled":False,"status":"disabled","referral_probability":None,"model_version":None}
    if not MODEL_PATH.exists():
        return {"enabled":False,"status":"not_trained","referral_probability":None,"model_version":None,"artifact_path":str(MODEL_PATH)}
    try:
        import joblib
        model=joblib.load(MODEL_PATH)
        import pandas as pd
        x=pd.DataFrame([[float(features.get(k,0)) for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
        probability=float(model.predict_proba(x)[0][1])
        meta=json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
        return {"enabled":True,"status":"loaded","referral_probability":round(probability,4),"model_version":meta.get("model_version","demo-logistic-v2"),"training_rows":meta.get("training_rows")}
    except Exception as exc:
        return {"enabled":False,"status":"error","error":str(exc),"referral_probability":None}
