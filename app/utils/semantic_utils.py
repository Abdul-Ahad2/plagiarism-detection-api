from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from app.config import (
        SEMANTIC_HIGH as CFG_HIGH,
        SEMANTIC_MED as CFG_MED,
        SEMANTIC_MIN as CFG_MIN,
        CHUNK_SIZE_WORDS as CFG_CHUNK,
        CHUNK_OVERLAP_WORDS as CFG_OVERLAP,
        TOPK_DOC_CANDIDATES as CFG_TOPK_DOCS,
        TOPK_CHUNK_MATCHES as CFG_TOPK_CHUNKS,
    )
except Exception:
    CFG_HIGH = 0.90
    CFG_MED = 0.85
    CFG_MIN = 0.83
    CFG_CHUNK = 250
    CFG_OVERLAP = 75
    CFG_TOPK_DOCS = 100
    CFG_TOPK_CHUNKS = 10

# ---------- Model Singleton ----------
_MODEL: Optional[SentenceTransformer] = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedder() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL

# ---------- Chunking ----------
def chunk_text(text: str, chunk_size_words: int = CFG_CHUNK, overlap_words: int = CFG_OVERLAP) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + chunk_size_words)
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(i + chunk_size_words - overlap_words, j)
    return chunks

# ---------- Embedding ----------
def embed_texts(texts: List[str], normalize: bool = True) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_embedder()
    vecs = model.encode(texts, normalize_embeddings=normalize)
    return np.array(vecs, dtype=np.float32)

def unit_norm(vecs: np.ndarray) -> np.ndarray:
    if vecs.size == 0:
        return vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms

# ---------- Similarity ----------
def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    return np.matmul(A, B.T).astype(np.float32)

def confidence_label(c: float) -> Literal["high", "medium", "low"]:
    if c >= CFG_HIGH: return "high"
    if c >= CFG_MED: return "medium"
    return "low"
