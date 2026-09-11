"""Train the corrective-RAG reranker after enough human feedback exists."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag.train_reranker import main
main()
