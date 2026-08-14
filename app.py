from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
from pydantic import BaseModel
from typing import Optional
import numpy as np
import json
import pickle
import re
import string
import os
import traceback

#  Path constants (edit these if your files live elsewhere)
RAW_DATA_PATH       = "data/raw/raw_data.json"
VECTORIZER_PATH     = "vectorizer.pkl"
TFIDF_MATRIX_PATH   = "tfidf_matrix.npz"
W2V_MODEL_PATH      = "models/embeddings/word2vec.model"
DOC_EMBEDDINGS_PATH = "models/embeddings/doc_embeddings_w2v.npy"


app = FastAPI(title="Intelligent Search Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Always return JSON, never a bare "Internal Server Error" string 
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"[ERROR] {exc}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
    )

# Global state
raw_docs         = []
tfidf_vectorizer = None
tfidf_matrix     = None
w2v_model        = None
doc_embeddings   = None

# Bootstrap 
@app.on_event("startup")
async def load_models():
    global raw_docs, tfidf_vectorizer, tfidf_matrix, w2v_model, doc_embeddings

    # Ensure NLTK data is present (silent no-ops if already downloaded)
    try:
        import nltk
        for pkg in ("punkt", "punkt_tab", "stopwords"):
            nltk.download(pkg, quiet=True)
        print("✓ NLTK data ready")
    except Exception as e:
        print(f"⚠  NLTK download failed: {e}")

    # Raw docs
    if os.path.exists(RAW_DATA_PATH):
        with open(RAW_DATA_PATH, "r") as f:
            raw_docs = json.load(f)
        print(f"✓ Loaded {len(raw_docs)} documents")
    else:
        print(f"⚠  raw_data.json not found at {RAW_DATA_PATH}")

    # TF-IDF
    try:
        import scipy.sparse as sp
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted
        with open(VECTORIZER_PATH, "rb") as f:
            candidate = pickle.load(f)
        check_is_fitted(candidate)  # raises NotFittedError if idf_ is absent
        tfidf_matrix = sp.load_npz(TFIDF_MATRIX_PATH)
        tfidf_vectorizer = candidate
        print(f"✓ TF-IDF model loaded  (vocab: {len(tfidf_vectorizer.vocabulary_):,} terms)")
    except NotFittedError:
        print("⚠  vectorizer.pkl is NOT fitted — re-run Baseline_1 and re-save the engine")
    except Exception as e:
        print(f"⚠  TF-IDF model not loaded: {e}")

    # Word2Vec
    try:
        from gensim.models import Word2Vec
        w2v_model = Word2Vec.load(W2V_MODEL_PATH)
        doc_embeddings = np.load(DOC_EMBEDDINGS_PATH)
        print("✓ Word2Vec model loaded")
    except Exception as e:
        print(f"⚠  Word2Vec model not loaded: {e}")


# Helpers 
def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tfidf_search(query: str, top_k: int = 5):
    if tfidf_vectorizer is None or tfidf_matrix is None:
        raise HTTPException(status_code=503, detail="TF-IDF model not loaded — check vectorizer.pkl and tfidf_matrix.npz")
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.exceptions import NotFittedError
        q_vec  = tfidf_vectorizer.transform([preprocess(query)])
        scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    except NotFittedError:
        raise HTTPException(
            status_code=503,
            detail="TF-IDF vectorizer is not fitted. Re-run Baseline_1 and copy the saved vectorizer.pkl.",
        )
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "rank":  int(i + 1),
            "title": raw_docs[idx]["title"],
            "snippet": raw_docs[idx]["text"][:250],
            "score": round(float(scores[idx]), 4),
        }
        for i, idx in enumerate(top_idx)
    ]


def _get_vector(tokens):
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(w2v_model.vector_size)


def w2v_search(query: str, top_k: int = 5):
    if w2v_model is None or doc_embeddings is None:
        raise HTTPException(status_code=503, detail="Word2Vec model not loaded")
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from sklearn.metrics.pairwise import cosine_similarity
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in word_tokenize(query.lower()) if w.isalnum() and w not in stop_words]
    q_vec  = _get_vector(tokens)
    scores = cosine_similarity([q_vec], doc_embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "rank":  int(i + 1),
            "title": raw_docs[idx]["title"],
            "snippet": raw_docs[idx]["text"][:250],
            "score": round(float(scores[idx]), 4),
        }
        for i, idx in enumerate(top_idx)
    ]


# Request/response models 
class SearchRequest(BaseModel):
    query: str
    model: str = "tfidf"   # "tfidf" | "w2v"
    top_k: int = 5

class CompareRequest(BaseModel):
    query: str
    top_k: int = 5


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return f.read()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "documents": len(raw_docs),
        "tfidf_loaded": tfidf_vectorizer is not None,
        "w2v_loaded":   w2v_model is not None,
    }


@app.post("/search")
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if req.model == "tfidf":
        results = tfidf_search(req.query, req.top_k)
    elif req.model == "w2v":
        results = w2v_search(req.query, req.top_k)
    else:
        raise HTTPException(status_code=400, detail="model must be 'tfidf' or 'w2v'")
    return {"query": req.query, "model": req.model, "results": results}


@app.post("/compare")
async def compare(req: CompareRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    tfidf_results = tfidf_search(req.query, req.top_k) if tfidf_vectorizer else []
    w2v_results   = w2v_search(req.query, req.top_k)   if w2v_model else []
    return {
        "query": req.query,
        "tfidf": tfidf_results,
        "w2v":   w2v_results,
    }
