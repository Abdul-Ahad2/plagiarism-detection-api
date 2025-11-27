from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from datetime import datetime
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient
import os

from app.config import MONGODB_URI,ALGORITHM, SECRET_KEY

from app.schemas.teacher_schemas import (
    TeacherLexicalBatchReport, TeacherLexicalSummary,
    LexicalDocResult, LexicalMatch
)
from app.utils.file_utils import extract_text_from_file, allowed_file
from app.utils.lexical_utils import (
    get_meaningful_sentences, extract_keywords,
    find_exact_matches, find_partial_phrase_match,
)
from app.utils.web_utils import fetch_sources, fetch_sources_multi_query

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
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    t0 = datetime.utcnow()
    doc_results: List[LexicalDocResult] = []
    total_matches = 0

    print(f"🔍 Starting teacher lexical analysis for {len(files)} uploaded file(s)...")

    for idx, f in enumerate(files, start=1):
        if not allowed_file(f.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {f.filename}")

        raw = await f.read()
        try:
            text = extract_text_from_file(raw, f.filename) or ""
        except ValueError as ve:
        # Catch over-word files
            raise HTTPException(status_code=400, detail=str(ve))
        sentences = get_meaningful_sentences(text)

        print(f"\n📄 Processing file {idx}: {f.filename}")
        print(f"   ➤ Extracted {len(sentences)} sentences")
        print(f"   ➤ Approx word count: {len(text.split())}")

        # Build search query from keywords
        sources = fetch_sources_multi_query(text, num_results=10)
        print(f"   ➤ Found {len(sources)} online sources from diverse queries")

        if not sources:
            raise HTTPException(status_code=404, detail=f"No sources found online for {f.filename}")

        matches: List[LexicalMatch] = []
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
                    matches.append(LexicalMatch(
                        matched_text=best_overall_match,
                        similarity=pct,
                        source_type=best_overall_src["type"],
                        source_title=best_overall_src["title"],
                        source_url=best_overall_src["source_url"],
                        section=None,
                        context="Potential plagiarism detected",
                    ))
                    source_matches_count[best_overall_src['source_url']] += 1
                    highest = max(highest, pct)
                    total_matches += 1
                    print(f"      ✅ Match ({pct}%) with {best_overall_src['source_url'][:50]}")

        # Better flagging logic considering multiple sources
        num_sources_with_matches = sum(1 for c in source_matches_count.values() if c > 0)
        avg_match_score = (sum(m.similarity for m in matches) / len(matches)) if matches else 0.0
        
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

        doc_results.append(LexicalDocResult(
            id=idx,
            name=f.filename,
            author=None,
            similarity=round(highest, 1),
            flagged=flagged,
            wordCount=len(text.split()),
            matches=matches,
            content=text  # Include full document for frontend
        ))

    highest_any = max((d.similarity for d in doc_results), default=0.0)
    avg = round(sum(d.similarity for d in doc_results) / len(doc_results), 1) if doc_results else 0.0
    flagged_count = sum(1 for d in doc_results if d.flagged)

    elapsed = (datetime.utcnow() - t0).total_seconds()
    mm = int(elapsed // 60)
    ss = int(elapsed % 60)
    processing = f"{mm}m {ss:02d}s"

    print("\n✅ Analysis completed!")
    print(f"   ➤ Total Documents: {len(doc_results)}")
    print(f"   ➤ Flagged: {flagged_count}")
    print(f"   ➤ Highest Similarity: {highest_any}%")
    print(f"   ➤ Average Similarity: {avg}%")
    print(f"   ➤ Processing Time: {processing}")

    result = TeacherLexicalBatchReport(
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

    # Save to MongoDB
    try:
        mongo_client = await get_mongo_client()
        db = mongo_client.sluethink
        reports_collection = db.reports
        
        # Extract unique sources from all matches
        all_sources = set()
        for doc in doc_results:
            for match in doc.matches:
                all_sources.add(match.source_url)
        
        # Prepare document for MongoDB
        report_doc = {
            "name": f"Batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "analysisType": "lexical",
            "submittedBy": current_user.get("username", "System"),
            "uploadDate": datetime.utcnow().strftime("%Y-%m-%d"),
            "similarity": highest_any,
            "status": "completed",
            "flagged": flagged_count > 0,
            "fileCount": len(doc_results),
            "processingTime": processing,
            "avgSimilarity": avg,
            "sources": list(all_sources),
            "createdAt": datetime.utcnow(),
            "userId": current_user.get("sub") or current_user.get("user_id"),
            # Store full analysis details
            "documents": [
                {
                    "id": doc.id,
                    "name": doc.name,
                    "similarity": doc.similarity,
                    "flagged": doc.flagged,
                    "wordCount": doc.wordCount,
                    "matchCount": len(doc.matches),
                    "matches": [
                        {
                            "matched_text": m.matched_text,
                            "similarity": m.similarity,
                            "source_url": m.source_url,
                            "source_title": m.source_title,
                            "source_type": m.source_type,
                        }
                        for m in doc.matches
                    ]
                }
                for doc in doc_results
            ],
            "summary": {
                "totalDocuments": result.summary.totalDocuments,
                "flaggedDocuments": result.summary.flaggedDocuments,
                "highestSimilarity": result.summary.highestSimilarity,
                "averageSimilarity": result.summary.averageSimilarity,
                "totalMatches": result.summary.totalMatches,
            }
        }
        
        # Insert into MongoDB
        insert_result = await reports_collection.insert_one(report_doc)
        print(f"\n💾 Report saved to MongoDB with ID: {insert_result.inserted_id}")
        
        # Update the result with the MongoDB ID
        result.id = str(insert_result.inserted_id)
        
        mongo_client.close()
        
    except Exception as e:
        print(f"\n❌ Error saving to MongoDB: {str(e)}")
        # Don't fail the request if MongoDB save fails
        # The analysis results are still returned

    print(f"\n🧾 Returning report:\n"
          f"  Total Docs: {result.summary.totalDocuments}\n"
          f"  Flagged Docs: {result.summary.flaggedDocuments}\n"
          f"  Avg Similarity: {result.summary.averageSimilarity}%\n"
          f"  Highest Similarity: {result.summary.highestSimilarity}%\n"
          f"  Total Matches: {result.summary.totalMatches}")

    return result