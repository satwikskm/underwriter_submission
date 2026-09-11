"""Train a lightweight retrieval reranker from human RAG feedback.

This intentionally refuses to train until there is enough feedback from both
positive and negative labels; it avoids manufacturing a misleading model.
"""
from pathlib import Path
import json, re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from app.rag.feedback import load_feedback
ROOT=Path(__file__).resolve().parents[2]
MODEL_PATH=ROOT/"data"/"rag_reranker.joblib"

def _text(row): return f"{row.get('query','')} {row.get('section','')} {row.get('source','')}"

def main():
    rows=load_feedback(); labels=[r.get("label") for r in rows]
    if len(rows)<20 or len(set(labels))<2:
        raise RuntimeError(f"Need at least 20 labeled feedback rows and both labels; found {len(rows)} rows and labels={set(labels)}")
    model=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),("clf",LogisticRegression(max_iter=1000,class_weight="balanced"))])
    model.fit([_text(r) for r in rows],labels)
    joblib.dump(model,MODEL_PATH)
    meta={"model_version":"rag-reranker-v1","training_rows":len(rows),"positive_rate":sum(labels)/len(labels)}
    (MODEL_PATH.with_suffix(".json")).write_text(json.dumps(meta,indent=2))
    print(json.dumps({"artifact":str(MODEL_PATH),**meta},indent=2))
if __name__=="__main__": main()
