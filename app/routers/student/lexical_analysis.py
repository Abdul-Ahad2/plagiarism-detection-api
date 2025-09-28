import re
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorClient
from jose import jwt, JWTError

from app.schemas.plagiarism_schemas import MatchDetail
from app.schemas.report_schemas import ReportDetail
from app.utils.file_utils import extract_text_from_file, allowed_file
from app.utils.lexical_utils import (
    normalize_text,
    get_meaningful_sentences,
    extract_keywords,
    find_exact_matches,
    find_partial_phrase_match,
)
from app.config import MONGODB_URI

router = APIRouter()

SECRET_KEY = "your_nextauth_secret"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_mongo_client():
    return AsyncIOMotorClient(MONGODB_URI)

@router.post("/student/lexical-analysis", response_model=ReportDetail)
async def check_plagiarism(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    mongo_client: AsyncIOMotorClient = Depends(get_mongo_client),
    token_payload: dict = Depends(verify_token),
):
    start = datetime.utcnow()
    db = mongo_client.get_default_database()
    reports_collection = db["reports"]
    data_collection = db["datas"]  

    # 1) Extract raw text
    if file and file.filename:
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type.")
        content_bytes = await file.read()
        raw_text = extract_text_from_file(content_bytes, file.filename)
        title = file.filename
    elif text and text.strip():
        raw_text = text
        title = "pasted_text_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    else:
        raise HTTPException(status_code=400, detail="No file or text provided.")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No readable text found.")

    # 2) Sentence split
    sentences = get_meaningful_sentences(raw_text)
    if not sentences:
        empty_doc = {
            "user_id": token_payload.get("sub"),
            "name": title,
            "content": raw_text,
            "date": datetime.utcnow(),
            "similarity": 0.0,
            "sources": [],
            "word_count": len(raw_text.split()),
            "time_spent": "00:00",
            "flagged": False,
            "plagiarism_data": [],
        }
        insert_result = await reports_collection.insert_one(empty_doc)
        return ReportDetail(
            id=str(insert_result.inserted_id),
            name=title,
            content=raw_text,
            plagiarism_data=[]
        )

    # 3) Build query
    keywords = extract_keywords(raw_text, max_keywords=5)
    query = " ".join(keywords) if keywords else raw_text[:100]

    # 4) Fetch candidates from DB (prefer $text)
    external_texts: List[dict] = []
    docs = []
    try:
        cursor = data_collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}, "title": 1, "text": 1, "source_url": 1, "type": 1},
        ).sort([("score", {"$meta": "textScore"})]).limit(200)
        docs = await cursor.to_list(length=200)
    except Exception:
        pass

    if not docs:
        tokens = keywords or re.findall(r"\w+", query)
        if tokens:
            regex = "|".join(re.escape(t) for t in tokens)
            cursor = data_collection.find(
                {"$or": [
                    {"title": {"$regex": regex, "$options": "i"}},
                    {"text":  {"$regex": regex, "$options": "i"}},
                ]},
                {"title": 1, "text": 1, "source_url": 1, "type": 1}
            ).limit(200)
            docs = await cursor.to_list(length=200)

    for doc in docs:
        txt = (doc.get("text") or "").strip()
        if txt:
            external_texts.append({
                "text": txt,
                "title": doc.get("title", "Unknown"),
                "source_url": doc.get("source_url", ""),
                "type": doc.get("type", "other"),
            })

    # 5) No candidates → empty report
    if not external_texts:
        empty_doc = {
            "user_id": token_payload.get("sub"),
            "name": title,
            "content": raw_text,
            "date": datetime.utcnow(),
            "similarity": 0.0,
            "sources": [],
            "word_count": len(raw_text.split()),
            "time_spent": "00:00",
            "flagged": False,
            "plagiarism_data": [],
        }
        insert_result = await reports_collection.insert_one(empty_doc)
        return ReportDetail(
            id=str(insert_result.inserted_id),
            name=title,
            content=raw_text,
            plagiarism_data=[]
        )

    # 6) Matching (uses improved lexical funcs)
    plagiarism_data_for_db: List[dict] = []
    highest_similarity = 0.0
    all_matched_titles = set()

    for orig in sentences:
        matched = False
        for ext in external_texts:
            sim = find_exact_matches(orig, ext["text"])
            if sim is not None:
                score = round(sim, 3)
                plagiarism_data_for_db.append({
                    "matched_text": orig,
                    "similarity":   score,
                    "source_type":  ext["type"],
                    "source_title": ext["title"],
                    "source_url":   ext["source_url"],
                })
                matched = True
                highest_similarity = max(highest_similarity, sim)
                all_matched_titles.add(ext["title"])
                break

        if not matched:
            for ext in external_texts:
                partial = find_partial_phrase_match(orig, ext["text"])
                if partial:
                    phrase, sim = partial
                    score = round(sim, 3)
                    plagiarism_data_for_db.append({
                        "matched_text": phrase,
                        "similarity":   score,
                        "source_type":  ext["type"],
                        "source_title": ext["title"],
                        "source_url":   ext["source_url"],
                    })
                    highest_similarity = max(highest_similarity, sim)
                    all_matched_titles.add(ext["title"])
                    break

    # 7) Time & save
    elapsed = datetime.utcnow() - start
    total_sec = int(elapsed.total_seconds())
    mins, secs = divmod(total_sec, 60)
    hours, mins = divmod(mins, 60)
    time_spent = f"{hours:d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"

    highest_pct = round(highest_similarity * 100, 1)
    flagged = highest_pct > 70

    report_doc = {
        "user_id": token_payload.get("sub"),
        "name": title,
        "date": datetime.utcnow(),
        "similarity": highest_pct,
        "sources": list(all_matched_titles),
        "word_count": len(raw_text.split()),
        "time_spent": time_spent,
        "flagged": flagged,
        "plagiarism_data": [
            {
                "matched_text": e["matched_text"],
                "similarity":   e["similarity"],
                "source_type":  e["source_type"],
                "source_title": e["source_title"],
                "source_url":   e["source_url"],
            }
            for e in plagiarism_data_for_db
        ],
    }
    print(report_doc)
    insert_res = await reports_collection.insert_one(report_doc)
    return ReportDetail(
        id=str(insert_res.inserted_id),
        name=title,
        content=raw_text,
        plagiarism_data=plagiarism_data_for_db
    )