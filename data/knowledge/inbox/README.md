# RAG knowledge inbox

Drop new underwriting `.md` or `.txt` documents here. Run:

```bash
python scripts/build_rag_index.py
```

The retriever will persist a local TF-IDF vector index under `data/rag_index.joblib`.
For production, replace this local index with pgvector, Qdrant, OpenSearch, or another managed vector store.
