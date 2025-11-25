from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Optional
from datetime import datetime
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient
import os

from app.config import MONGODB_URI,ALGORITHM, SECRET_KEY
from app.schemas.teacher_schemas import (
    LexicalMatch
)
from app.utils.file_utils import extract_text_from_file, allowed_file
from app.utils.lexical_utils import (
    get_meaningful_sentences, extract_keywords,
    find_exact_matches, find_partial_phrase_match,
)
from app.utils.web_utils import fetch_sources, fetch_sources_multi_query

router = APIRouter(prefix="/student", tags=["student-lexical"])

LEXICAL_DOC_THRESHOLD = 0.85  # 85%
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")



def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_mongo_client():
    return AsyncIOMotorClient(MONGODB_URI)

@router.post("/lexical-analysis")
async def student_lexical_analysis(
    file: UploadFile = File(...),
    current_user=Depends(verify_token),
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    t0 = datetime.utcnow()
    total_matches = 0

    print(f"🔍 Starting student lexical analysis for uploaded file...")

    # Process single file
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

    raw = await file.read()
    text = extract_text_from_file(raw, file.filename) or ""
    sentences = get_meaningful_sentences(text)

    print(f"\n📄 Processing file: {file.filename}")
    print(f"   ➤ Extracted {len(sentences)} sentences")
    print(f"   ➤ Approx word count: {len(text.split())}")

    # Build search query from keywords
    sources = fetch_sources_multi_query(text, num_results=10)
    print(f"   ➤ Found {len(sources)} online sources from diverse queries")

    if not sources:
        raise HTTPException(status_code=404, detail=f"No sources found online for {file.filename}")

    matches = []
    highest = 0.0
    source_matches_count = {}

    externals = [
        {
            "title": s.get("url", "Unknown"),
            "text": s.get("content", ""),
            "source_url": s.get("url", ""),
            "type": "web",
        }
        for s in sources if s.get("content")
    ]

    for ext in externals:
        print(f"      🌐 Source: {ext['source_url'][:60]}...")
        source_matches_count[ext['source_url']] = 0

    # Compare each sentence against ALL sources
    for s in sentences:
        best_overall_score = 0.0
        best_overall_match = None
        best_overall_src = None

        for ext in externals:
            # Try exact match first
            sim = find_exact_matches(s, ext["text"])
            if sim is not None and sim > best_overall_score:
                best_overall_score = sim
                best_overall_match = s
                best_overall_src = ext
                continue

            # Try partial phrase match
            pp = find_partial_phrase_match(s, ext["text"])
            if pp:
                phrase, score = pp
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_match = phrase
                    best_overall_src = ext

        # Add match if found and above threshold (50%)
        if best_overall_match and best_overall_score > 0.0:
            pct = round(best_overall_score * 100.0, 1)
            
            if pct >= 50:
                matches.append({
                    "matched_text": best_overall_match,
                    "similarity": pct,
                    "source_type": best_overall_src["type"],
                    "source_title": best_overall_src["title"],
                    "source_url": best_overall_src["source_url"],
                    "context": "Potential plagiarism detected",
                })
                source_matches_count[best_overall_src['source_url']] += 1
                highest = max(highest, pct)
                total_matches += 1
                print(f"      ✅ Match ({pct}%) with {best_overall_src['source_url'][:50]}")

    # Better flagging logic considering multiple sources
    num_sources_with_matches = sum(1 for c in source_matches_count.values() if c > 0)
    avg_match_score = (sum(m["similarity"] for m in matches) / len(matches)) if matches else 0.0
    
    # Flag if any of these conditions are met:
    # 1. Single source with high similarity (>85%)
    # 2. Content plagiarized from 2+ different sources
    # 3. 3+ matches with average >70%
    flagged = (
        highest >= 85 or
        num_sources_with_matches >= 2 or
        (len(matches) >= 3 and avg_match_score >= 70)
    )
    
    print(f"   ➤ Highest similarity: {highest:.1f}%")
    print(f"   ➤ Total matches: {len(matches)}")
    print(f"   ➤ Sources with matches: {num_sources_with_matches}")
    print(f"   ➤ Average match score: {avg_match_score:.1f}%")
    print(f"   ➤ Flagged: {flagged}")

    elapsed = (datetime.utcnow() - t0).total_seconds()
    mm = int(elapsed // 60)
    ss = int(elapsed % 60)
    processing_time = f"{mm}m {ss:02d}s"

    print("\n✅ Analysis completed!")
    print(f"   ➤ Flagged: {flagged}")
    print(f"   ➤ Highest Similarity: {highest}%")
    print(f"   ➤ Average Similarity: {avg_match_score:.1f}%")
    print(f"   ➤ Processing Time: {processing_time}")

    # Extract unique sources
    all_sources = list(set(m["source_url"] for m in matches))

    # Build response
    result = {
        "id": None,  # Will be set after MongoDB insert
        "name": file.filename,
        "content": text,
        "matches": matches,
        "similarity": round(highest, 1),
        "flagged": flagged,
        "wordCount": len(text.split()),
        "processingTime": processing_time,
        "totalMatches": total_matches,
        "averageSimilarity": round(avg_match_score, 1),
        "sources": all_sources,
        "uploadDate": datetime.utcnow().isoformat(),
    }

    # Save to MongoDB
    try:
        mongo_client = await get_mongo_client()
        db = mongo_client.sluethink
        reports_collection = db.reports
        
        # Prepare document for MongoDB
        report_doc = {
            "name": file.filename,
            "analysisType": "lexical",
            "submittedBy": current_user.get("username", "System"),
            "uploadDate": datetime.utcnow().strftime("%Y-%m-%d"),
            "similarity": highest,
            "status": "completed",
            "flagged": flagged,
            "fileCount": 1,
            "processingTime": processing_time,
            "avgSimilarity": avg_match_score,
            "sources": all_sources,
            "createdAt": datetime.utcnow(),
            "userId": current_user.get("sub") or current_user.get("user_id"),
            "content": text,
            "wordCount": len(text.split()),
            "matches": matches,
            "totalMatches": total_matches,
        }
        
        # Insert into MongoDB
        insert_result = await reports_collection.insert_one(report_doc)
        print(f"\n💾 Report saved to MongoDB with ID: {insert_result.inserted_id}")
        
        # Update the result with the MongoDB ID
        result["id"] = str(insert_result.inserted_id)
        
        mongo_client.close()
        
    except Exception as e:
        print(f"\n❌ Error saving to MongoDB: {str(e)}")
        # Don't fail the request if MongoDB save fails
        result["id"] = "temp_id"

    print(f"\n🧾 Returning report:\n"
          f"  Flagged: {flagged}\n"
          f"  Avg Similarity: {avg_match_score:.1f}%\n"
          f"  Highest Similarity: {highest}%\n"
          f"  Total Matches: {total_matches}")

    return result