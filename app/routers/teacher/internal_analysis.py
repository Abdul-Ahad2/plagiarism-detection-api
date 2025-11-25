from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Tuple, Set
from datetime import datetime
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient

from app.schemas.teacher_schemas import (
    DocumentInfo, OverlapDetail, ComparisonDetail,
    InternalReportDetail, InternalReportSummary
)
from app.utils.file_utils import extract_text_from_file, allowed_file
from app.utils.lexical_utils import (
    find_partial_phrase_match_for_internal,
    get_meaningful_sentences,
    find_exact_matches,
    find_partial_phrase_match,
)
from app.config import MONGODB_URI,ALGORITHM, SECRET_KEY

router = APIRouter(prefix="/teacher", tags=["teacher-internal"])

LEXICAL_PAIR_THRESHOLD = 0.50  # 50% - pairs above this are flagged
OVERLAP_MIN_TOKENS = 12

# Add these new thresholds for color coding:
HIGH_SIMILARITY_THRESHOLD = 0.85   # 85% - Red (very high)
MEDIUM_SIMILARITY_THRESHOLD = 0.70  # 70% - Yellow (medium)
LOW_SIMILARITY_THRESHOLD = 0.50  
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
    return max((o.similarity for o in overlaps), default=0.0)


def _create_overlap_key(name_a: str, name_b: str, text: str, similarity: float, context: str) -> str:
    """Create unique key for overlap deduplication - includes context to distinguish different match types"""
    # Normalize text to handle whitespace variations
    text_normalized = ' '.join(text.split())
    return f"{name_a}|{name_b}|{text_normalized}|{similarity}|{context}"


def _extract_matched_text_from_sentence(sent_b: str, phrase: str) -> str:
    """Extract the actual text from sent_b that matches the phrase"""
    if not sent_b or not phrase:
        return phrase
    
    # Normalize both for comparison
    phrase_normalized = ' '.join(phrase.split()).lower()
    sent_normalized = ' '.join(sent_b.split()).lower()
    sent_b_normalized = ' '.join(sent_b.split())  # Keep original casing
    
    # If phrase exists in sentence, extract it as-is from original
    if phrase_normalized in sent_normalized:
        start_idx = sent_normalized.find(phrase_normalized)
        end_idx = start_idx + len(phrase_normalized)
        return sent_b_normalized[start_idx:end_idx].strip()
    
    # If not found exactly, try to find similar chunks
    # Split into words and try to find the best match
    phrase_words = phrase_normalized.split()
    sent_words = sent_normalized.split()
    
    # Look for the phrase words in the sentence
    for i in range(len(sent_words) - len(phrase_words) + 1):
        if sent_words[i:i+len(phrase_words)] == phrase_words:
            return ' '.join(sent_b_normalized.split()[i:i+len(phrase_words)])
    
    # Fallback: return the phrase as-is
    return phrase


def _find_overlaps_for_pair(
    name_a: str, sents_a: List[str],
    name_b: str, sents_b: List[str],
    seen_overlaps: Set[str]
) -> List[OverlapDetail]:
    """Find all overlaps between two document's sentences"""
    overlaps: List[OverlapDetail] = []
    
    for sent_a in sents_a:
        # Check exact matches
        for sent_b in sents_b:
            exact_score = find_exact_matches(sent_a, sent_b)
            if exact_score is not None:
                sim_pct = _percent(exact_score)
                if sim_pct >= LEXICAL_PAIR_THRESHOLD * 100:
                    context = "Exact/near-exact sentence overlap"
                    overlap_key = _create_overlap_key(name_a, name_b, sent_a, sim_pct, context)
                    if overlap_key not in seen_overlaps:
                        seen_overlaps.add(overlap_key)
                        overlaps.append(OverlapDetail(
                            fromDoc=name_a, 
                            toDoc=name_b, 
                            text=sent_a,
                            similarity=sim_pct,
                            sectionA=sent_a,
                            sectionB=sent_b,
                            context=context,
                        ))
        
        # Check partial phrase matches
        best_partial = None
        best_score = 0.0
        best_sent_b = None
        
        for sent_b in sents_b:
            partial_result = find_partial_phrase_match_for_internal(sent_a, sent_b)
            if partial_result:
                phrase, score = partial_result
                print(f"DEBUG: Partial match - phrase: {phrase[:80]}, score: {score}")
                if score > best_score:
                    best_score = score
                    best_partial = phrase
                    best_sent_b = sent_b
        
        # Add best partial match if it meets threshold
        if best_partial and best_sent_b and len(best_partial.split()) >= OVERLAP_MIN_TOKENS:
            sim_pct = _percent(best_score)
            if sim_pct >= LEXICAL_PAIR_THRESHOLD * 100:
                context = "High-overlap phrase (shingle/containment)"
                overlap_key = _create_overlap_key(name_a, name_b, best_partial, sim_pct, context)
                if overlap_key not in seen_overlaps:
                    seen_overlaps.add(overlap_key)
                    overlaps.append(OverlapDetail(
                        fromDoc=name_a, 
                        toDoc=name_b, 
                        text=best_partial,
                        similarity=sim_pct,
                        sectionA=sent_a,
                        sectionB=best_sent_b,
                        context=context,
                    ))
    
    return overlaps

@router.post("/internal-analysis", response_model=InternalReportDetail)
async def internal_analysis(
    files: List[UploadFile] = File(...),
    token_payload: dict = Depends(verify_token),
    mongo: AsyncIOMotorClient = Depends(get_mongo_client),
):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Upload at least 2 files")

    t0 = datetime.utcnow()

    # --- Load & sentence-split all docs ---
    docs: List[Tuple[str, List[str]]] = []
    doc_infos: List[DocumentInfo] = []
    doc_texts = {}

    for idx, f in enumerate(files, start=1):
        if not allowed_file(f.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {f.filename}")
        raw = await f.read()
        text = extract_text_from_file(raw, f.filename) or ""
        sents = get_meaningful_sentences(text)
        doc_infos.append(DocumentInfo(id=idx, name=f.filename, author=None))
        docs.append((f.filename, sents))
        doc_texts[f.filename] = text

    # --- Pairwise comparisons ---
    comparisons: List[ComparisonDetail] = []
    seen_overlaps: Set[str] = set()

    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            name_a, sents_a = docs[i]
            name_b, sents_b = docs[j]
            
            # Find all overlaps for this pair
            overlaps = _find_overlaps_for_pair(
                name_a, sents_a,
                name_b, sents_b,
                seen_overlaps
            )

            # Calculate pair score and flag if needed
            pair_score = _aggregate_pair_score(overlaps)
            flagged = pair_score >= LEXICAL_PAIR_THRESHOLD * 100
            
            comp = ComparisonDetail(
                id=_ordered_pair_key(i + 1, j + 1),
                docA=name_a,
                docB=name_b,
                similarity=round(pair_score, 1),
                flagged=flagged,
                overlaps=overlaps,
                contentA=doc_texts[name_a],
                contentB=doc_texts[name_b],
            )
            if flagged:
                comparisons.append(comp)

    # --- Compute per-document results ---
    doc_results = []
    total_matches = 0
    flagged_count = 0

    for d_idx, d in enumerate(doc_infos, start=1):
        name = d.name
        word_count = len(doc_texts[name].split())
        matches = [o for c in comparisons for o in c.overlaps if o.fromDoc == name or o.toDoc == name]
        highest_similarity = max((o.similarity for o in matches), default=0.0)
        flagged = highest_similarity >= LEXICAL_PAIR_THRESHOLD * 100
        if flagged:
            flagged_count += 1
        total_matches += len(matches)

        doc_results.append({
            "id": d.id,
            "name": d.name,
            "similarity": round(highest_similarity, 1),
            "flagged": flagged,
            "wordCount": word_count,
            "matchCount": len(matches),
            "matches": matches
        })

    highest_any = max(d['similarity'] for d in doc_results) if doc_results else 0.0
    avg_similarity = round(sum(d['similarity'] for d in doc_results) / len(doc_results), 1) if doc_results else 0.0
    elapsed = (datetime.utcnow() - t0).total_seconds()
    processing = f"{int(elapsed // 60)}m {int(elapsed % 60):02d}s"

    report = InternalReportDetail(
        id="internal_report",
        name="Internal Plagiarism Check",
        uploadDate=datetime.utcnow(),
        processingTime=processing,
        documents=doc_infos,
        comparisons=comparisons,
        summary=InternalReportSummary(
            totalDocuments=len(doc_results),
            totalComparisons=(len(docs) * (len(docs) - 1)) // 2,
            flaggedComparisons=flagged_count,
            highestSimilarity=round(highest_any, 1),
            averageSimilarity=avg_similarity,
        ),
    )

    # --- Save to MongoDB ---
    try:
        db = mongo.sluethink
        reports_collection = db.reports

        all_sources = set()
        for comp in comparisons:
            for o in comp.overlaps:
                all_sources.add(o.toDoc)

        report_doc = {
            "name": f"Internal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "analysisType": "internal",
            "submittedBy": token_payload.get("name", "System"),
            "uploadDate": datetime.utcnow().strftime("%Y-%m-%d"),
            "similarity": highest_any,
            "status": "completed",
            "flagged": flagged_count > 0,
            "fileCount": len(doc_results),
            "processingTime": processing,
            "avgSimilarity": avg_similarity,
            "totalMatches": total_matches,
            "sources": list(all_sources),
            "createdAt": datetime.utcnow(),
            "userId": token_payload.get("sub") or token_payload.get("user_id"),
            "documents": [
                {
                    "id": d['id'],
                    "name": d['name'],
                    "similarity": d['similarity'],
                    "flagged": d['flagged'],
                    "wordCount": d['wordCount'],
                    "matchCount": d['matchCount'],
                    "matches": [
                        {
                            "matched_text": m.text,
                            "similarity": m.similarity,
                            "source_url": m.toDoc,
                            "source_title": m.toDoc,
                            "source_type": "internal",
                        } for m in d['matches']
                    ]
                } for d in doc_results
            ],
            "summary": {
                "totalDocuments": len(doc_results),
                "flaggedDocuments": flagged_count,
                "highestSimilarity": highest_any,
                "averageSimilarity": avg_similarity,
                "totalMatches": total_matches,
            }
        }

        insert_result = await reports_collection.insert_one(report_doc)
        print(f"💾 Report saved to MongoDB with ID: {insert_result.inserted_id}")
        report.id = str(insert_result.inserted_id)

    except Exception as e:
        print(f"❌ Error saving to MongoDB: {str(e)}")

    return report