from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from datetime import datetime
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient
import logging
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from app.config import MONGODB_URI, ALGORITHM, SECRET_KEY
from app.schemas.teacher_schemas import (
    TeacherLexicalBatchReport, TeacherLexicalSummary,
    LexicalDocResult, LexicalMatch
)
from app.utils.file_utils import extract_text_from_file, allowed_file
from app.utils.semantic_utils import (
    generate_five_queries,
    find_semantic_matches,
)
from app.utils.web_utils import fetch_sources_multi_query
from app.utils.ai_detector import detect_ai_similarity

router = APIRouter(prefix="/teacher", tags=["teacher-semantic"])

SEMANTIC_THRESHOLD = 0.50
SCRAPING_TIMEOUT = 180  # 3 minutes total for all queries combined

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

class ScrapingTimeoutManager:
    """Manages web scraping with hard 3-minute overall timeout"""
    
    def __init__(self, timeout_seconds: int = 180):
        self.timeout = timeout_seconds
        self.start_time = None
        self.sources = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        self.cancelled = False
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def is_timeout(self) -> bool:
        """Check if 3-minute timeout exceeded"""
        return self.elapsed() >= self.timeout
    
    async def fetch_all_queries(self, queries: List[str], num_results: int = 5) -> List:
        """
        Fetch sources for all queries with hard 180-second overall timeout.
        Immediately stops and starts matching when timeout reached.
        """
        self.start_time = datetime.utcnow()
        self.sources = []
        
        logger.info(f"\n🔎 WEB SCRAPING PHASE")
        logger.info(f"   Max Duration: {self.timeout}s (3 minutes)")
        logger.info(f"   Queries: {len(queries)}")
        logger.info(f"   Starting: {self.start_time.strftime('%H:%M:%S')}")
        
        # Process all queries in parallel with timeout
        tasks = []
        for query_idx, query in enumerate(queries, 1):
            logger.info(f"\n   Query {query_idx}/{len(queries)}: {query[:60]}...")
            tasks.append(self._fetch_query(query, num_results))
        
        try:
            # Wait for all tasks with overall timeout
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"\n🛑 HARD TIMEOUT REACHED after {self.elapsed():.1f}s")
            logger.warning(f"   Cancelling all pending queries")
            self.cancelled = True
            # Cancel remaining tasks
            for task in tasks:
                if isinstance(task, asyncio.Task):
                    task.cancel()
        
        # Remove duplicates
        seen_urls = set()
        unique_sources = []
        for source in self.sources:
            url = source.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        elapsed = self.elapsed()
        logger.info(f"\n✅ SCRAPING PHASE STOPPED")
        logger.info(f"   Total Duration: {elapsed:.1f}s ({int(elapsed)//60}m {int(elapsed)%60}s)")
        logger.info(f"   Unique Sources: {len(unique_sources)}")
        logger.info(f"   Status: {'🛑 TIMEOUT' if self.is_timeout() else '✅ COMPLETED'}")
        
        return unique_sources
    
    async def _fetch_query(self, query: str, num_results: int = 5):
        """Fetch sources for a single query"""
        try:
            sources = await asyncio.to_thread(
                fetch_sources_multi_query,
                query,
                num_results
            )
            
            with self.lock:
                self.sources.extend(sources)
            
            logger.info(f"      ✅ Found {len(sources)} sources")
            
        except asyncio.CancelledError:
            logger.warning(f"      ⏭️  Query cancelled (timeout)")
        except Exception as e:
            logger.error(f"      ❌ Error: {e}")
    
    def cleanup(self):
        """Clean up executor"""
        try:
            self.executor.shutdown(wait=False)
        except:
            pass

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

        # ✅ AI DETECTION
        logger.info(f"🤖 Running AI detection...")
        ai_similarity = detect_ai_similarity(text)
        logger.info(f"   AI Similarity: {ai_similarity}")

        # Generate 3 semantic queries
        queries = generate_five_queries(text)
        
        # ✅ WEB SCRAPING WITH 3-MINUTE HARD TIMEOUT (OVERALL)
        scraper = ScrapingTimeoutManager(timeout_seconds=SCRAPING_TIMEOUT)
        try:
            unique_sources = await scraper.fetch_all_queries(queries, num_results=5)
        finally:
            scraper.cleanup()
        
        logger.info(f"   Total unique sources: {len(unique_sources)}")

        if not unique_sources:
            logger.warning(f"   ⚠️  No sources found, skipping semantic matching")
            doc_results.append(LexicalDocResult(
                id=idx,
                name=f.filename,
                author=None,
                similarity=0.0,
                flagged=False,
                wordCount=len(text.split()),
                matches=[],
                content=text[:5000],
                ai_similarity=ai_similarity
            ))
            continue

        # ✅ MATCHING PHASE (starts immediately after timeout)
        logger.info(f"\n📊 SEMANTIC MATCHING PHASE")
        logger.info(f"   Comparing against {len(unique_sources)} sources...")
        
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

        for ext_idx, ext in enumerate(externals, 1):
            logger.info(f"      Source {ext_idx}/{len(externals)}: {ext['source_url'][:60]}...")
            source_matches_count[ext['source_url']] = 0

            try:
                # Semantic comparison
                semantic_matches = find_semantic_matches(
                    text,
                    ext["text"],
                    threshold=SEMANTIC_THRESHOLD
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
            
            except Exception as e:
                logger.error(f"         Error matching source: {e}")
                continue

        # Deduplicate matches
        logger.info(f"   🔄 Deduplicating {len(matches)} matches...")
        unique_matches_dict = {}
        
        for match in matches:
            key = match.matched_text.lower().strip()
            if key not in unique_matches_dict or match.similarity > unique_matches_dict[key].similarity:
                unique_matches_dict[key] = match
        
        matches = list(unique_matches_dict.values())
        logger.info(f"   ✅ Deduplicated to {len(matches)} unique matches")

        # Recalculate metrics
        highest = max((m.similarity for m in matches), default=0.0)
        source_matches_count = {}
        for match in matches:
            source_matches_count[match.source_url] = source_matches_count.get(match.source_url, 0) + 1
        
        # Flagging logic
        num_sources_with_matches = sum(1 for c in source_matches_count.values() if c > 0)
        avg_match_score = (sum(m.similarity for m in matches) / len(matches)) if matches else 0.0
        
        flagged = (
            highest >= 80 or
            num_sources_with_matches >= 2 or
            (len(matches) >= 2 and avg_match_score >= 70)
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
            content=text[:5000],
            ai_similarity=ai_similarity
        ))

    # Final summary
    highest_any = max((d.similarity for d in doc_results), default=0.0)
    avg = round(sum(d.similarity for d in doc_results) / len(doc_results), 1) if doc_results else 0.0
    flagged_count = sum(1 for d in doc_results if d.flagged)
    avg_ai_similarity = round(sum(d.ai_similarity for d in doc_results) / len(doc_results), 3) if doc_results else 0.0

    elapsed = (datetime.utcnow() - t0).total_seconds()
    mm = int(elapsed // 60)
    ss = int(elapsed % 60)
    processing = f"{mm}m {ss:02d}s"

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"  Documents: {len(doc_results)}")
    logger.info(f"  Flagged: {flagged_count}")
    logger.info(f"  Highest Semantic Similarity: {highest_any}%")
    logger.info(f"  Average Semantic Similarity: {avg}%")
    logger.info(f"  Average AI Similarity: {avg_ai_similarity}")
    logger.info(f"  Total Matches: {total_matches}")
    logger.info(f"  Total Time: {processing}\n")

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
            averageAiSimilarity=avg_ai_similarity,
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
            "aiSimilarity": avg_ai_similarity,
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
                    "aiSimilarity": doc.ai_similarity,
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
                "averageAiSimilarity": result.summary.averageAiSimilarity,
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