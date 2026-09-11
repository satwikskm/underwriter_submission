"""Persistent local RAG + corrective retrieval (CRAG-style) without a heavyweight vector DB.

The index is a persisted TF-IDF vector matrix. New Markdown/text knowledge is
added under data/knowledge and indexed with scripts/build_rag_index.py.
"""
from pathlib import Path
import json, re
from functools import lru_cache
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn): return fn
        return decorator

ROOT=Path(__file__).resolve().parents[2]
KB_DIR=ROOT/"data"/"knowledge"
INDEX_PATH=ROOT/"data"/"rag_index.joblib"
RAG_TOP_K=int(__import__("os").getenv("RAG_TOP_K","4"))
MIN_RELEVANCE=float(__import__("os").getenv("RAG_MIN_RELEVANCE","0.08"))

STOPWORDS={"the","a","an","and","or","of","to","for","in","on","with","is","are","what","how","should","underwriting"}
EXPANSIONS={
    "claim":"claims loss history incurred loss", "claims":"claim loss history incurred",
    "loss":"claims loss ratio incurred", "construction":"building construction protection",
    "occupancy":"occupancy operations exposure", "sprinkler":"sprinkler protection fire suppression",
    "refer":"referral escalation underwriting", "referral":"refer escalation underwriting",
    "property":"commercial property building", "fire":"fire protection sprinkler hazard"
}

def _documents_from_source():
    docs=[]
    for p in sorted(KB_DIR.rglob("*")):
        if p.suffix.lower() not in {".md",".txt"} or p.name.startswith("README"):
            continue
        text=p.read_text(encoding="utf-8",errors="ignore")
        lines=text.splitlines()
        document_title=next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("# ")), p.stem)
        blocks=[b.strip() for b in re.split(r"(?=^## )",text,flags=re.M) if b.strip()]
        for block in blocks:
            if not block.startswith("## "):
                continue
            title=block.splitlines()[0].lstrip("# ").strip()
            body=block.split("\n",1)[1].strip() if "\n" in block else ""
            if not body: continue
            docs.append({"source":str(p.relative_to(ROOT)),"section":title,"text":f"{document_title}\n{block}"})
    return docs

def build_index():
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs=_documents_from_source()
    if not docs: raise RuntimeError(f"No .md/.txt knowledge documents found under {KB_DIR}")
    vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,2),sublinear_tf=True)
    matrix=vectorizer.fit_transform([d["text"] for d in docs])
    import joblib
    payload={"version":"rag-tfidf-v2","vectorizer":vectorizer,"matrix":matrix,"docs":docs}
    joblib.dump(payload,INDEX_PATH)
    _index.cache_clear()
    return {"index_path":str(INDEX_PATH),"documents":len(docs)}

@lru_cache(maxsize=1)
def _index():
    import joblib
    if not INDEX_PATH.exists():
        build_index()
    payload=joblib.load(INDEX_PATH)
    return payload["vectorizer"],payload["matrix"],payload["docs"]

def _retrieve_once(query:str,top_k:int)->list[dict]:
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer,matrix,docs=_index()
    q=vectorizer.transform([query])
    scores=cosine_similarity(q,matrix)[0]
    ranked=scores.argsort()[::-1][:top_k]
    return [{**docs[i],"relevance":round(float(scores[i]),4),"retrieval_query":query} for i in ranked if scores[i]>=MIN_RELEVANCE]

def _correct_query(query:str)->str:
    tokens=re.findall(r"[a-z0-9_]+",query.lower())
    expanded=list(tokens)
    for token in tokens:
        expanded.extend(EXPANSIONS.get(token,"" ).split())
    return " ".join(dict.fromkeys(expanded))

@traceable(name="underwriting.rag_retriever",run_type="retriever")
def corrective_retrieve(query:str,top_k:int=RAG_TOP_K)->dict:
    first=_retrieve_once(query,top_k)
    best=max((x["relevance"] for x in first),default=0.0)
    if first and best>=MIN_RELEVANCE:
        return {"results":first,"corrected":False,"original_query":query,"corrected_query":None,"quality":best,"status":"accepted"}
    corrected_query=_correct_query(query)
    second=_retrieve_once(corrected_query,top_k) if corrected_query!=query else []
    second_best=max((x["relevance"] for x in second),default=0.0)
    chosen=second if second_best>best else first
    return {"results":chosen,"corrected":bool(second and second_best>best),"original_query":query,"corrected_query":corrected_query if second and second_best>best else None,"quality":max(best,second_best),"status":"corrected" if second and second_best>best else ("no_evidence" if not chosen else "weak_evidence")}

@traceable(name="underwriting.rag_retriever_legacy",run_type="retriever")
def retrieve(query:str,top_k:int=RAG_TOP_K)->list[dict]:
    return corrective_retrieve(query,top_k)["results"]

def knowledge_stats()->dict:
    docs=_documents_from_source()
    return {"source_documents":len(docs),"indexed":INDEX_PATH.exists(),"index_path":str(INDEX_PATH)}
