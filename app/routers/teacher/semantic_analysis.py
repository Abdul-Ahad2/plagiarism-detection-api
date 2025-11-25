from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from datetime import datetime
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient
import logging

from app.config import MONGODB_URI, ALGORITHM, SECRET_KEY
from app.schemas.teacher_schemas import (
    TeacherLexicalBatchReport, TeacherLexicalSummary,
    LexicalDocResult, LexicalMatch
)
from app.utils.file_utils import extract_text_from_file, allowed_file
from app.utils.semantic_utils import (
    generate_three_queries,
    find_semantic_matches,
)
from app.utils.web_utils import fetch_sources_multi_query

router = APIRouter(prefix="/teacher", tags=["teacher-semantic"])

# ✅ THRESHOLD: 0.50 cosine similarity catches more paraphrasing
SEMANTIC_THRESHOLD = 0.50

# ✅ NEW: 500 word limit per document
MAX_DOCUMENT_WORDS = 500
WARN_DOCUMENT_WORDS = 400

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_analysis")

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_mongo_client():
    return AsyncIOMotorClient(MONGODB_URI)

@router.post("/semantic-analysis", response_model=TeacherLexicalBatchReport)
async def teacher_semantic_analysis(
    files: List[UploadFile] = File(...),
    current_user=Depends(verify_token),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    t0 = datetime.utcnow()
    doc_results: List[LexicalDocResult] = []
    total_matches = 0

    logger.info(f"\n{'='*80}")
    logger.info(f"🧠 SEMANTIC ANALYSIS - {len(files)} file(s)")
    logger.info(f"{'='*80}")

    for idx, f in enumerate(files, start=1):
        if not allowed_file(f.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {f.filename}")

        raw = await f.read()
        text = extract_text_from_file(raw, f.filename) or ""
        
        logger.info(f"\n📄 File {idx}: {f.filename}")
        logger.info(f"   Words: {len(text.split())}")

        # Generate 3 semantic queries
        queries = generate_three_queries(text)
        
        # Search web using generated queries
        logger.info(f"   🔎 Searching web with {len(queries)} queries...")
        all_sources = []
        for query_idx, query in enumerate(queries, 1):
            logger.info(f"      Query {query_idx}: {query[:60]}...")
            try:
                sources = fetch_sources_multi_query(query, num_results=5)
                logger.info(f"         Found {len(sources)} sources")
                all_sources.extend(sources)
            except Exception as e:
                logger.error(f"         Error: {e}")
        
        # Remove duplicate sources
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            url = source.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        logger.info(f"   ✅ Unique sources: {len(unique_sources)}")

        if not unique_sources:
            logger.warning(f"   ⚠️  No sources found")
            continue

        matches: List[LexicalMatch] = []
        highest = 0.0
        source_matches_count = {}

        # Prepare externals
        externals = [
            {
                "title": s.get("url", "Unknown"),
                "text": s.get("content", ""),
                "source_url": s.get("url", ""),
                "type": "web",
            }
            for s in unique_sources if s.get("content")
        ]

        logger.info(f"   📊 Comparing against {len(externals)} sources...")

        for ext_idx, ext in enumerate(externals, 1):
            logger.info(f"      Source {ext_idx}: {ext['source_url'][:60]}...")
            source_matches_count[ext['source_url']] = 0

            # Semantic comparison
            semantic_matches = find_semantic_matches(
                text,
                ext["text"],
                threshold=SEMANTIC_THRESHOLD  # ✅ Using lowered threshold
            )
            
            logger.info(f"         Found {len(semantic_matches)} semantic matches")
            
            for match in semantic_matches:
                similarity_pct = round(match['similarity'] * 100, 1)
                
                matches.append(LexicalMatch(
                    matched_text=match['doc_text'][:300],
                    similarity=similarity_pct,
                    source_type=ext["type"],
                    source_title=ext["title"],
                    source_url=ext["source_url"],
                    section=None,
                    context="Semantic similarity detected (possible paraphrasing)",
                ))
                
                source_matches_count[ext['source_url']] += 1
                highest = max(highest, similarity_pct)
                total_matches += 1
                
                logger.debug(f"            Match: {similarity_pct}% - {match['doc_text'][:50]}...")

        # ✅ NEW: Deduplicate matches - keep only highest similarity per unique text
        logger.info(f"   🔄 Deduplicating {len(matches)} matches...")
        unique_matches_dict = {}
        
        for match in matches:
            # Use matched text as key (normalized)
            key = match.matched_text.lower().strip()
            
            # Keep only if this is the highest similarity for this text
            if key not in unique_matches_dict or match.similarity > unique_matches_dict[key].similarity:
                unique_matches_dict[key] = match
        
        matches = list(unique_matches_dict.values())
        logger.info(f"   ✅ Deduplicated to {len(matches)} unique matches")

        # Recalculate highest and average with deduplicated matches
        highest = max((m.similarity for m in matches), default=0.0)
        source_matches_count = {}
        for match in matches:
            source_matches_count[match.source_url] = source_matches_count.get(match.source_url, 0) + 1
        
        # Flagging logic
        num_sources_with_matches = sum(1 for c in source_matches_count.values() if c > 0)
        avg_match_score = (sum(m.similarity for m in matches) / len(matches)) if matches else 0.0
        
        # ✅ IMPROVED: Better flagging thresholds
        flagged = (
            highest >= 80 or  # ✅ Lowered from 85 to 80
            num_sources_with_matches >= 2 or
            (len(matches) >= 2 and avg_match_score >= 70)  # ✅ Lowered from 3 matches to 2
        )
        
        logger.info(f"   📈 Results:")
        logger.info(f"      Highest: {highest:.1f}%")
        logger.info(f"      Total matches: {len(matches)}")
        logger.info(f"      Sources with matches: {num_sources_with_matches}")
        logger.info(f"      Average: {avg_match_score:.1f}%")
        logger.info(f"      Flagged: {flagged}")

        doc_results.append(LexicalDocResult(
            id=idx,
            name=f.filename,
            author=None,
            similarity=round(highest, 1),
            flagged=flagged,
            wordCount=len(text.split()),
            matches=matches,
            content=text[:5000]  # Store limited content
        ))

    highest_any = max((d.similarity for d in doc_results), default=0.0)
    avg = round(sum(d.similarity for d in doc_results) / len(doc_results), 1) if doc_results else 0.0
    flagged_count = sum(1 for d in doc_results if d.flagged)

    elapsed = (datetime.utcnow() - t0).total_seconds()
    mm = int(elapsed // 60)
    ss = int(elapsed % 60)
    processing = f"{mm}m {ss:02d}s"

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"  Documents: {len(doc_results)}")
    logger.info(f"  Flagged: {flagged_count}")
    logger.info(f"  Highest: {highest_any}%")
    logger.info(f"  Average: {avg}%")
    logger.info(f"  Total Matches: {total_matches}")
    logger.info(f"  Time: {processing}\n")

    result = TeacherLexicalBatchReport(
        id="teacher_semantic_batch",
        name="Teacher Semantic Analysis",
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
        
        all_sources = set()
        for doc in doc_results:
            for match in doc.matches:
                all_sources.add(match.source_url)
        
        report_doc = {
            "name": f"Semantic_Batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "analysisType": "semantic",
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
        
        insert_result = await reports_collection.insert_one(report_doc)
        logger.info(f"💾 Saved to MongoDB: {insert_result.inserted_id}")
        result.id = str(insert_result.inserted_id)
        mongo_client.close()
        
    except Exception as e:
        logger.error(f"❌ MongoDB error: {str(e)}")

    return result