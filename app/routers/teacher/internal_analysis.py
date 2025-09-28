from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Tuple
from datetime import datetime
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError,jwt
from motor.motor_asyncio import AsyncIOMotorClient

# from app.dependencies.auth import get_current_user
from app.routers.student.lexical_analysis import ALGORITHM, SECRET_KEY
from app.schemas.teacher_schemas import (
    DocumentInfo, OverlapDetail, ComparisonDetail,
    InternalReportDetail, InternalReportSummary
)
from app.utils.file_utils import extract_text_from_file, allowed_file
from app.utils.lexical_utils import (
    get_meaningful_sentences,
    find_exact_matches,
    find_partial_phrase_match,
)
from app.config import MONGODB_URI

router = APIRouter(prefix="/teacher", tags=["teacher-internal"])

LEXICAL_PAIR_THRESHOLD = 0.85  # 85% (percent scale handled below)
OVERLAP_MIN_TOKENS = 12
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_mongo_client():
    return AsyncIOMotorClient(MONGODB_URI)

def _percent(x: float) -> float:
    return round(float(x) * 100.0, 1)

def _ordered_pair_key(i: int, j: int) -> str:
    a, b = (i, j) if i < j else (j, i)
    return f"{a}-{b}"

def _aggregate_pair_score(overlaps: List[OverlapDetail]) -> float:
    # Use max overlap as aggregate for strict plagiarism signaling.
    # (If you prefer mean of top-k, swap logic here.)
    return max((o.similarity for o in overlaps), default=0.0)

@router.post("/internal-analysis", response_model=InternalReportDetail)
async def internal_analysis(
    files: List[UploadFile] = File(...),
    token_payload: dict = Depends(verify_token),
    mongo: AsyncIOMotorClient = Depends(get_mongo_client),
):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Upload at least 2 files")

    t0 = datetime.utcnow()

    # 1) Load & sentence-split all docs
    docs: List[Tuple[str, List[str]]] = []  # (name, sentences)
    doc_infos: List[DocumentInfo] = []
    for idx, f in enumerate(files, start=1):
        if not allowed_file(f.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {f.filename}")
        raw = await f.read()
        text = extract_text_from_file(raw, f.filename) or ""
        sents = get_meaningful_sentences(text)
        doc_infos.append(DocumentInfo(id=idx, name=f.filename, author=None))
        docs.append((f.filename, sents))

    # 2) Pairwise comparisons
    comparisons: List[ComparisonDetail] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            nameA, sentsA = docs[i]
            nameB, sentsB = docs[j]

            overlaps: List[OverlapDetail] = []

            # Simple symmetric pass (A→B; if needed also B→A)
            for s in sentsA:
                # exact/full-ish
                exact = None
                for t in sentsB:
                    exact = find_exact_matches(s, t)
                    if exact is not None:
                        sim_pct = _percent(exact)
                        if sim_pct >= LEXICAL_PAIR_THRESHOLD * 100:
                            overlaps.append(OverlapDetail(
                                fromDoc=nameA, toDoc=nameB, text=s,
                                similarity=sim_pct,
                                sectionA=None, sectionB=None,
                                context="Exact/near-exact sentence overlap",
                            ))
                        break  # take first exact-ish and move on
                if exact is not None:
                    continue

                # partial phrase
                best_partial = None
                best_score = 0.0
                for t in sentsB:
                    partial = find_partial_phrase_match(s, t)
                    if partial:
                        phrase, score = partial
                        if score > best_score:
                            best_score = score
                            best_partial = phrase
                if best_partial:
                    sim_pct = _percent(best_score)
                    if sim_pct >= LEXICAL_PAIR_THRESHOLD * 100:
                        # token length guard
                        if len(best_partial.split()) >= OVERLAP_MIN_TOKENS:
                            overlaps.append(OverlapDetail(
                                fromDoc=nameA, toDoc=nameB, text=best_partial,
                                similarity=sim_pct,
                                sectionA=None, sectionB=None,
                                context="High-overlap phrase (shingle/containment)",
                            ))

            pair_score = _aggregate_pair_score(overlaps)
            flagged = pair_score >= LEXICAL_PAIR_THRESHOLD * 100
            comp = ComparisonDetail(
                id=_ordered_pair_key(i + 1, j + 1),
                docA=nameA,
                docB=nameB,
                similarity=round(pair_score, 1),
                flagged=flagged,
                overlaps=overlaps,
            )
            if flagged:
                comparisons.append(comp)

    # 3) Summary
    highest = max((c.similarity for c in comparisons), default=0.0)
    avg = round(sum(c.similarity for c in comparisons) / len(comparisons), 1) if comparisons else 0.0

    elapsed = (datetime.utcnow() - t0).total_seconds()
    mm = int(elapsed // 60)
    ss = int(elapsed % 60)
    processing = f"{mm}m {ss:02d}s"

    report = InternalReportDetail(
        id="internal_report",
        name="Internal Plagiarism Check",
        uploadDate=datetime.utcnow(),
        processingTime=processing,
        documents=doc_infos,
        comparisons=comparisons,
        summary=InternalReportSummary(
            totalDocuments=len(docs),
            totalComparisons=(len(docs) * (len(docs) - 1)) // 2,
            flaggedComparisons=len(comparisons),
            highestSimilarity=round(highest, 1),
            averageSimilarity=avg,
        ),
    )
    return report
