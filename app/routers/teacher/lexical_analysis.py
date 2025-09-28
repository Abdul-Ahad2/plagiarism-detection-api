from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from datetime import datetime
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError,jwt
from motor.motor_asyncio import AsyncIOMotorClient

# from app.dependencies.auth import get_current_user
from app.routers.student.lexical_analysis import ALGORITHM, SECRET_KEY
from app.schemas.teacher_schemas import (
    TeacherLexicalBatchReport, TeacherLexicalSummary,
    LexicalDocResult, LexicalMatch
)
from app.utils.file_utils import extract_text_from_file, allowed_file
from app.utils.lexical_utils import (
    get_meaningful_sentences, extract_keywords,
    find_exact_matches, find_partial_phrase_match,
)
from app.config import MONGODB_URI

router = APIRouter(prefix="/teacher", tags=["teacher-lexical"])

LEXICAL_DOC_THRESHOLD = 0.85  # 85%
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_mongo_client():
    return AsyncIOMotorClient(MONGODB_URI)

@router.post("/lexical-analysis", response_model=TeacherLexicalBatchReport)
async def teacher_lexical_analysis(
    files: List[UploadFile] = File(...),
    current_user=Depends(verify_token),
    mongo: AsyncIOMotorClient = Depends(get_mongo_client),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    t0 = datetime.utcnow()
    db = mongo.get_default_database()
    data_collection = db["datas"]

    doc_results: List[LexicalDocResult] = []
    total_matches = 0

    for idx, f in enumerate(files, start=1):
        if not allowed_file(f.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {f.filename}")
        raw = await f.read()
        text = extract_text_from_file(raw, f.filename) or ""
        sentences = get_meaningful_sentences(text)

        # candidate fetch (text index first; fallback to regex with keywords)
        keywords = extract_keywords(text, max_keywords=5)
        query = " ".join(keywords) if keywords else (text[:100] or f.filename)

        docs = []
        try:
            cursor = data_collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "title": 1, "text": 1, "source_url": 1, "type": 1},
            ).sort([("score", {"$meta": "textScore"})]).limit(200)
            docs = await cursor.to_list(length=200)
        except Exception:
            pass

        if not docs and keywords:
            import re as _re
            regex = "|".join(_re.escape(k) for k in keywords)
            cursor = data_collection.find(
                {"$or": [
                    {"title": {"$regex": regex, "$options": "i"}},
                    {"text":  {"$regex": regex, "$options": "i"}},
                ]},
                {"title": 1, "text": 1, "source_url": 1, "type": 1},
            ).limit(200)
            docs = await cursor.to_list(length=200)

        # matching
        matches: List[LexicalMatch] = []
        highest = 0.0

        externals = [
            {
                "title": d.get("title", "Unknown"),
                "text": (d.get("text") or "").strip(),
                "source_url": d.get("source_url", ""),
                "type": d.get("type", "other"),
            }
            for d in docs if (d.get("text") or "").strip()
        ]

        for s in sentences:
            hit = False
            # exact
            for ext in externals:
                sim = find_exact_matches(s, ext["text"])
                if sim is not None:
                    pct = round(sim * 100.0, 1)
                    matches.append(LexicalMatch(
                        matched_text=s, similarity=pct,
                        source_type=ext["type"], source_title=ext["title"], source_url=ext["source_url"],
                        section=None, context="Exact/near-exact sentence overlap",
                    ))
                    highest = max(highest, pct)
                    total_matches += 1
                    hit = True
                    break
            if hit:
                continue

            # partial
            best_phrase = None
            best_score = 0.0
            best_src = None
            for ext in externals:
                pp = find_partial_phrase_match(s, ext["text"])
                if pp:
                    phrase, score = pp
                    if score > best_score:
                        best_score = score
                        best_phrase = phrase
                        best_src = ext
            if best_phrase:
                pct = round(best_score * 100.0, 1)
                matches.append(LexicalMatch(
                    matched_text=best_phrase, similarity=pct,
                    source_type=best_src["type"], source_title=best_src["title"], source_url=best_src["source_url"],
                    section=None, context="High-overlap phrase (shingle/containment)",
                ))
                highest = max(highest, pct)
                total_matches += 1

        flagged = highest >= LEXICAL_DOC_THRESHOLD * 100
        doc_results.append(LexicalDocResult(
            id=idx, name=f.filename, author=None,
            similarity=round(highest, 1), flagged=flagged,
            wordCount=len(text.split()), matches=matches
        ))

    highest_any = max((d.similarity for d in doc_results), default=0.0)
    avg = round(sum(d.similarity for d in doc_results) / len(doc_results), 1) if doc_results else 0.0
    flagged_count = sum(1 for d in doc_results if d.flagged)

    elapsed = (datetime.utcnow() - t0).total_seconds()
    mm = int(elapsed // 60); ss = int(elapsed % 60)
    processing = f"{mm}m {ss:02d}s"

    return TeacherLexicalBatchReport(
        id="teacher_lexical_batch",
        name="Teacher Lexical Analysis",
        uploadDate=datetime.utcnow(),
        processingTime=processing,
        documents=doc_results,
        summary=TeacherLexicalSummary(
            totalDocuments=len(doc_results),
            flaggedDocuments=flagged_count,
            highestSimilarity=highest_any,
            averageSimilarity=avg,
            totalMatches=total_matches,
        ),
    )
