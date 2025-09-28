from __future__ import annotations
from typing import List, Tuple, Optional
from datetime import datetime
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from motor.motor_asyncio import AsyncIOMotorClient

from app.routers.student.lexical_analysis import ALGORITHM, SECRET_KEY
from app.schemas.teacher_schemas import (
    DocumentInfo,
    SemanticOverlap,
    SemanticComparison,
    TeacherSemanticReport,
    InternalReportSummary,
)
from app.utils.file_utils import extract_text_from_file, allowed_file
from app.utils.semantic_utils import (
    chunk_text,
    unit_norm,
    cosine_sim_matrix,
    confidence_label,
    get_embedder,
)
from app.config import MONGODB_URI

# Local HF pipeline for RAG generation
from transformers import pipeline
import re

router = APIRouter(prefix="/teacher", tags=["teacher-semantic"])

# Expect header: Authorization: Bearer <token>
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# ---- Auth ----
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        # NOTE: this expects HS256-signed token compatible with your SECRET_KEY/ALGORITHM
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_mongo_client():
    return AsyncIOMotorClient(MONGODB_URI)

# ---- RAG LLM ----
_GENERATOR: Optional[object] = None

def get_generator():
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6", 
            device=-1  # CPU; set to 0 for GPU
        )
    return _GENERATOR

def build_rag_context(comparisons: List[SemanticComparison], externals: List[dict]) -> str:
    """Build context from retrieved documents for RAG"""
    context_parts = []
    
    for comp in comparisons:
        for overlap in comp.overlaps[:2]:  # Top 2 overlaps per comparison
            context_parts.append(f"Match found with '{overlap.textB}' at {overlap.cosine_pct:.1f}% similarity")
    
    return ". ".join(context_parts[:5])  # Limit to 5 matches

def create_rag_prompt(doc_names: str, comparisons: List[SemanticComparison]) -> str:
    """
    Build a clear, structured prompt so the summarizer reliably produces
    one sentence per uploaded doc + a final overall conclusion.
    """
    # collect top fact per document
    lines = []
    for comp in comparisons:
        if not comp.overlaps:
            continue
        top = comp.overlaps[0]
        lines.append(
            f"- Document '{comp.docA}' → Source '{top.textB}' • {top.cosine_pct:.1f}% ({top.confidence})"
        )

    if not lines:
        lines = ["- No strong overlaps detected."]

    matches_block = "\n".join(lines)

    # instruct exactly what we want (paragraph, 3–6 sentences, no bullets)
    return (
        f"Documents analyzed: {doc_names}\n\n"
        f"Top semantic matches (per document):\n{matches_block}\n\n"
        "Write a short teacher-style plagiarism narrative as a single paragraph, 3–6 sentences total.\n"
        "Requirements:\n"
        "• Write ONE sentence per uploaded document that states which source it most closely matches and the similarity (with confidence).\n"
        "• End with ONE concluding sentence that summarizes how many documents were flagged and the highest similarity observed.\n"
        "• Use fluent, formal English. Do NOT use bullet points. Do NOT repeat these instructions."
    )

def _per_doc_facts(comparisons: List[SemanticComparison]) -> List[dict]:
    facts = []
    for comp in comparisons:
        if comp.overlaps:
            o = comp.overlaps[0]
            facts.append({
                "doc": comp.docA,
                "src": o.textB,
                "pct": float(o.cosine_pct),
                "conf": o.confidence
            })
    return facts

INSTRUCTION_PATTERNS = [
    r"\bwrite (a )?(short )?(teacher[- ]style)?\b.*?(paragraph|sentences?)\b",
    r"\b(3|3–6|3-6)[-–]?\s*sentences?\b",
    r"\bone sentence per\b",
    r"\b(conclude|end) with\b.*",
    r"\bdo not (use|list|repeat)\b.*",
    r"\brequirements?:\b.*",
    r"\bdocuments analyzed:\b.*?$",
    r"\bsimilarity findings:\b.*?$",
    r"^[•\-\*]\s.*?$",                         # bullet lines
]


def cleanup_llm_text(text: str) -> str:
    import re
    t = (text or "").strip()

    # drop obvious instruction echoes (line by line)
    lines = [ln.strip() for ln in re.split(r"\s*\n\s*|\s{2,}", t)]
    kept = []
    for ln in lines:
        drop = any(re.search(pat, ln, flags=re.IGNORECASE) for pat in INSTRUCTION_PATTERNS)
        if not drop:
            kept.append(ln)
    t = " ".join(kept)

    # normalize spaces / dangling punctuation
    t = re.sub(r"\s+", " ", t).replace(" .", ".").replace(" ,", ",").strip()

    # split to sentences, remove very short or imperative “write …”
    sents = re.split(r"(?<=[.!?])\s+", t)
    sents = [s.strip() for s in sents if len(s.strip()) >= 10 and not re.match(r"(?i)^write\b", s.strip())]

    if len(sents) > 6:
        sents = sents[:6]
    if not sents:
        return ""

    t = " ".join(sents)
    if not t.endswith("."):
        t += "."
    return t




def generate_rag_response(
    report_id: str,
    docs: List[DocumentInfo],
    comparisons: List[SemanticComparison],
    externals: List[dict]
) -> str:
    if not comparisons:
        return "No significant similarities were found."

    generator = get_generator()
    doc_names = ", ".join(d.name for d in docs) or "N/A"
    prompt = create_rag_prompt(doc_names, comparisons)

    try:
        result = generator(
            prompt,
            max_length=700,
            min_length=80,
            do_sample=False,
            early_stopping=True
        )
        text = result[0]["summary_text"]

        # --- post-process to guarantee coverage + nice tone ---
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text.endswith("."):
            text += "."
        
        text = cleanup_llm_text(text)

        facts = _per_doc_facts(comparisons)
        lower_text = text.lower()
        for f in facts:
            doc_label = f["doc"]
            if doc_label.lower() not in lower_text:
                text += (
            f" The document '{f['doc']}' shows a strong semantic overlap with "
            f"'{f['src']}' ({f['pct']:.1f}%, {f['conf']})."
        )


        # ensure a concluding sentence exists (flagged count + highest similarity)
        if not re.search(r"(overall|in total|highest similarity|highest match)", text, re.IGNORECASE):
            flagged = sum(1 for c in comparisons if c.flagged)
            highest = max((c.similarity for c in comparisons), default=0.0)
            text += (
                f" In total, {flagged} document(s) were flagged; the highest observed similarity was "
                f"{highest:.1f}%. These findings indicate substantial semantic overlap that may reflect "
                f"paraphrasing or AI-assisted rewriting."
            )

        # final polish: normalize spacing/casing quirks
        text = text.replace(" .", ".").replace(" ,", ",")
        return text

    except Exception as e:
        print("LLM error:", e)
        return "Analysis completed but automatic narrative generation failed."

    

# ---- Main route ----
@router.post("/semantic-analysis", response_model=TeacherSemanticReport)
async def teacher_semantic_analysis(
    files: List[UploadFile] = File(...),
    current_user=Depends(verify_token),
    mongo: AsyncIOMotorClient = Depends(get_mongo_client),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    start_time = datetime.utcnow()
    texts: List[Tuple[str, str]] = []
    docs: List[DocumentInfo] = []

    # 1) Extract teacher-uploaded docs
    for idx, f in enumerate(files, start=1):
        if not allowed_file(f.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {f.filename}")
        raw = await f.read()
        txt = extract_text_from_file(raw, f.filename) or ""
        texts.append((f.filename, txt))
        docs.append(DocumentInfo(id=idx, name=f.filename))

    # 2) Load external corpus (with embeddings) - These are our RAG documents
    db = mongo.get_default_database()
    data_collection = db["datas"]
    cursor = data_collection.find({}, {"title": 1, "content": 1, "embedding": 1}).limit(5000)
    externals = await cursor.to_list(length=5000)

    ext_titles = [e.get("title", "Unknown") for e in externals]
    ext_vecs = [np.array(e.get("embedding") or [], dtype=np.float32) for e in externals]

    # keep only valid-size vectors (384 for all-MiniLM-L6-v2)
    idx_map = [k for k, v in enumerate(ext_vecs) if v.size == 384]
    ext_vecs = (
        np.stack([unit_norm(np.array(ext_vecs[k]).reshape(1, -1))[0] for k in idx_map], axis=0)
        if idx_map else np.zeros((0, 384), dtype=np.float32)
    )

    if ext_vecs.shape[0] == 0:
        raise HTTPException(status_code=500, detail="No valid external embeddings found in corpus")

    # 3) Retrieval: Find semantically similar documents
    comparisons: List[SemanticComparison] = []
    embedder = get_embedder()

    for i, (nameA, txtA) in enumerate(texts):
        chunksA = chunk_text(txtA)
        if not chunksA:
            continue

        vecA = embedder.encode(chunksA, normalize_embeddings=True)
        sims = cosine_sim_matrix(vecA, ext_vecs)

        overlaps: List[SemanticOverlap] = []
        # Flatten and take best-scoring pairs first (Retrieval step)
        flat = np.dstack(np.unravel_index(np.argsort(-sims, axis=None), sims.shape))[0]

        for ii, jj in flat[:10]:
            c = float(sims[ii, jj])
            if c < 0.83:
                break
            ext_idx = idx_map[jj]
            overlaps.append(
                SemanticOverlap(
                    textA=chunksA[ii],
                    textB=ext_titles[ext_idx],
                    cosine=c,
                    cosine_pct=round(c * 100.0, 1),
                    confidence=confidence_label(c),
                )
            )

        if overlaps:
            agg = round(max(o.cosine_pct for o in overlaps), 1)
            comparisons.append(
                SemanticComparison(
                    id=f"{i+1}-ext",
                    docA=nameA,
                    docB="external_corpus",
                    similarity=agg,
                    flagged=agg >= 85.0,
                    overlaps=overlaps,
                )
            )

    # 4) Stats
    highest = max((c.similarity for c in comparisons), default=0.0)
    avg = round(sum(c.similarity for c in comparisons) / len(comparisons), 1) if comparisons else 0.0
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    mm, ss = divmod(int(elapsed), 60)
    processing = f"{mm}m {ss:02d}s"

    # 5) Generation: RAG-based response using retrieved documents
    narrative = generate_rag_response("teacher_semantic", docs, comparisons, externals)

    # 6) Return as model
    return TeacherSemanticReport(
        id="teacher_semantic",
        name="Teacher Semantic Analysis",
        analysisType="semantic",
        mode="external",
        uploadDate=datetime.utcnow(),
        processingTime=processing,
        documents=docs,
        comparisons=comparisons,
        summary=InternalReportSummary(
            totalDocuments=len(docs),
            totalComparisons=len(docs),
            flaggedComparisons=len(comparisons),
            highestSimilarity=highest,
            averageSimilarity=avg,
        ),
        narrative=narrative,
    )