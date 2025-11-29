from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from datetime import datetime
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient
import logging
import asyncio
import threading

from app.config import MONGODB_URI, ALGORITHM, SECRET_KEY
from app.schemas.teacher_schemas import (
    TeacherLexicalBatchReport, TeacherLexicalSummary,
    LexicalDocResult, LexicalMatch
)
from app.utils.file_utils import extract_text_from_file, allowed_file
from app.utils.lexical_utils import (
    get_meaningful_sentences, extract_keywords,
    find_exact_matches, find_partial_phrase_match,
)
from app.utils.web_utils import fetch_sources_multi_query

router = APIRouter(prefix="/teacher", tags=["teacher-lexical"])

LEXICAL_DOC_THRESHOLD = 0.85  # 85%

# ✅ HARD TIMEOUT: 3 minutes (180 seconds) for all queries combined
SCRAPING_TIMEOUT = 180

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lexical_analysis")

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_mongo_client():
    return AsyncIOMotorClient(MONGODB_URI)

def generate_five_queries(text: str) -> List[str]:
    """
    Generate 5 high-quality search queries from document.
    Covers: beginning, 1/4, middle, 3/4, end
    """
    from app.utils.lexical_utils import get_meaningful_sentences
    
    logger.info("   🔍 Generating 5 lexical queries from content...")
    
    sentences = get_meaningful_sentences(text)
    if len(sentences) < 5:
        logger.warning("   ⚠️  Not enough sentences, using fewer queries")
        # Fallback for short documents
        words = text.split()
        return [
            ' '.join(words[:30]) if len(words) > 0 else text,
            ' '.join(words[max(0, len(words)//4):max(0, len(words)//4)+30]) if len(words) > 30 else text,
            ' '.join(words[max(0, len(words)//2):max(0, len(words)//2)+30]) if len(words) > 30 else text,
        ]
    
    queries = []
    
    # ✅ Query 1: BEGINNING - First 3-4 sentences
    beginning_end = min(4, len(sentences))
    query1 = ' '.join(sentences[:beginning_end])
    queries.append(query1)
    logger.debug(f"   Query 1 length: {len(query1.split())} words")
    
    # ✅ Query 2: QUARTER-POINT - Around 25% of document
    quarter_start = max(beginning_end, len(sentences) // 4)
    quarter_end = min(quarter_start + 4, len(sentences))
    query2 = ' '.join(sentences[quarter_start:quarter_end])
    queries.append(query2)
    logger.debug(f"   Query 2 length: {len(query2.split())} words")
    
    # ✅ Query 3: MIDDLE - Around 50% of document
    mid_start = max(quarter_end, len(sentences) // 2)
    mid_end = min(mid_start + 4, len(sentences))
    query3 = ' '.join(sentences[mid_start:mid_end])
    queries.append(query3)
    logger.debug(f"   Query 3 length: {len(query3.split())} words")
    
    # ✅ Query 4: THREE-QUARTER-POINT - Around 75% of document
    three_quarter_start = max(mid_end, int(len(sentences) * 0.75))
    three_quarter_end = min(three_quarter_start + 4, len(sentences))
    query4 = ' '.join(sentences[three_quarter_start:three_quarter_end])
    queries.append(query4)
    logger.debug(f"   Query 4 length: {len(query4.split())} words")
    
    # ✅ Query 5: END - Last 3-4 sentences
    end_start = max(three_quarter_end, len(sentences) - 4)
    query5 = ' '.join(sentences[end_start:])
    queries.append(query5)
    logger.debug(f"   Query 5 length: {len(query5.split())} words")
    
    # ✅ Validate queries
    final_queries = []
    for q in queries:
        q = q.strip()
        if len(q.split()) >= 15:  # Minimum 15 words for good search
            final_queries.append(q)
    
    logger.info(f"   ✅ Generated {len(final_queries)} queries:")
    for i, q in enumerate(final_queries, 1):
        word_count = len(q.split())
        preview = q[:80] + "..." if len(q) > 80 else q
        logger.info(f"      Query {i} ({word_count} words): {preview}")
    
    return final_queries

class ScrapingTimeoutManager:
    """Manages web scraping with hard 3-minute overall timeout"""
    
    def __init__(self, timeout_seconds: int = 180):
        self.timeout = timeout_seconds
        self.start_time = None
        self.sources = []
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
    
    async def fetch_all_sources(self, queries: List[str], num_results: int = 10) -> List:
        """
        Fetch sources for all 5 queries with hard 180-second overall timeout.
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
    
    async def _fetch_query(self, query: str, num_results: int = 10):
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

    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 LEXICAL ANALYSIS - {len(files)} file(s)")
    logger.info(f"{'='*80}")

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

        logger.info(f"\n📄 File {idx}: {f.filename}")
        logger.info(f"   Sentences: {len(sentences)}")
        logger.info(f"   Words: {len(text.split())}")

        # ✅ Generate 5 lexical queries
        queries = generate_five_queries(text)
        
        # ✅ WEB SCRAPING WITH 3-MINUTE HARD TIMEOUT (OVERALL)
        scraper = ScrapingTimeoutManager(timeout_seconds=SCRAPING_TIMEOUT)
        sources = await scraper.fetch_all_sources(queries, num_results=5)
        
        # ✅ RESET TIMEOUT - Scraping phase is done, matching has no time limit
        from app.utils import web_utils
        web_utils._scraping_deadline = None
        web_utils._scraping_start_time = None
        
        logger.info(f"   Total unique sources: {len(sources)}")

        if not sources:
            logger.warning(f"   ⚠️  No sources found, skipping lexical matching")
            doc_results.append(LexicalDocResult(
                id=idx,
                name=f.filename,
                author=None,
                similarity=0.0,
                flagged=False,
                wordCount=len(text.split()),
                matches=[],
                content=text
            ))
            continue

        matches: List[LexicalMatch] = []
        highest = 0.0
        source_matches_count = {}

        # ✅ MATCHING PHASE (starts immediately after timeout)
        logger.info(f"\n📊 LEXICAL MATCHING PHASE")
        logger.info(f"   Comparing {len(sentences)} sentences against {len(sources)} sources...")

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
            logger.info(f"      🌐 Source: {ext['source_url'][:60]}...")
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
                    logger.debug(f"      ✅ Match ({pct}%) with {best_overall_src['source_url'][:50]}")

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
        
        logger.info(f"   📈 Results:")
        logger.info(f"      Highest similarity: {highest:.1f}%")
        logger.info(f"      Total matches: {len(matches)}")
        logger.info(f"      Sources with matches: {num_sources_with_matches}")
        logger.info(f"      Average match score: {avg_match_score:.1f}%")
        logger.info(f"      Flagged: {flagged}")

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

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"  Documents: {len(doc_results)}")
    logger.info(f"  Flagged: {flagged_count}")
    logger.info(f"  Highest: {highest_any}%")
    logger.info(f"  Average: {avg}%")
    logger.info(f"  Total Matches: {total_matches}")
    logger.info(f"  Total Time: {processing}\n")

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
            "name": f"Lexical_Batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
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
        logger.info(f"💾 Report saved to MongoDB with ID: {insert_result.inserted_id}")
        
        # Update the result with the MongoDB ID
        result.id = str(insert_result.inserted_id)
        
        mongo_client.close()
        
    except Exception as e:
        logger.error(f"❌ Error saving to MongoDB: {str(e)}")

    logger.info(f"\n🧾 Returning report:")
    logger.info(f"   Total Docs: {result.summary.totalDocuments}")
    logger.info(f"   Flagged Docs: {result.summary.flaggedDocuments}")
    logger.info(f"   Avg Similarity: {result.summary.averageSimilarity}%")
    logger.info(f"   Highest Similarity: {result.summary.highestSimilarity}%")
    logger.info(f"   Total Matches: {result.summary.totalMatches}\n")

    return result