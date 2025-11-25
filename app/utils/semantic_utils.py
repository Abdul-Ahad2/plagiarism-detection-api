"""
Semantic analysis with MiniLM embeddings for paraphrasing detection.
Detects even heavily paraphrased content from LLMs like ChatGPT.
"""

import re
from typing import List, Dict
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_utils")

# Lazy load model to avoid multiple loads
_encoder_model = None

def get_encoder():
    """Lazy load SentenceTransformer MiniLM model"""
    global _encoder_model
    if _encoder_model is None:
        logger.info("🔄 Loading SentenceTransformer MiniLM-L6-v2 model...")
        try:
            from sentence_transformers import SentenceTransformer
            _encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Model loaded successfully")
        except ImportError:
            logger.error("❌ sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
    return _encoder_model

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering out junk"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    cleaned = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 20 and len(sent.split()) >= 5:
            cleaned.append(sent)
    
    return cleaned

def extract_key_sentences(text: str, num_sentences: int = 5) -> List[str]:
    """
    Extract the most important sentences from text based on:
    - Position (first and last sentences often important)
    - Length (15-30 words is optimal)
    - Keyword frequency
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= num_sentences:
        return sentences
    
    scored_sentences = []
    
    for idx, sent in enumerate(sentences):
        score = 0.0
        word_count = len(sent.split())
        
        # Position score
        if idx < len(sentences) * 0.2 or idx > len(sentences) * 0.8:
            score += 0.3
        
        # Length score
        if 15 <= word_count <= 30:
            score += 0.3
        elif 10 <= word_count <= 40:
            score += 0.15
        
        # Keyword diversity
        words = set(sent.lower().split())
        common_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'be', 'it', 'this', 'that'}
        unique_words = len(words - common_words)
        score += min(unique_words / 10, 0.4)
        
        scored_sentences.append((score, sent))
    
    scored_sentences.sort(reverse=True)
    key_sents = [sent for _, sent in scored_sentences[:num_sentences]]
    
    result = []
    for sent in sentences:
        if sent in key_sents:
            result.append(sent)
    
    return result

def generate_three_queries(text: str, max_words: int = 3000) -> List[str]:
    """
    Generate 3 high-quality semantic search queries from document.
    Uses longer, more meaningful paragraphs for better web search results.
    
    Query 1: Main Topic Paragraph (beginning)
    Query 2: Supporting Evidence Paragraph (middle)
    Query 3: Conclusion/Related Concepts Paragraph (end)
    """
    logger.info("   🔍 Generating semantic queries from content...")
    
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
    
    sentences = split_into_sentences(text)
    if len(sentences) < 5:
        # Fallback for very short text
        logger.warning("   ⚠️  Very short document, using basic queries")
        return [
            ' '.join(words[:30]),
            ' '.join(words[max(0, len(words)//3):max(0, len(words)//3)+30]),
            ' '.join(words[max(0, 2*len(words)//3):max(0, 2*len(words)//3)+30])
        ]
    
    queries = []
    
    # ✅ Query 1: BEGINNING PARAGRAPH - First 3-4 sentences (main topic)
    # This sets up the main subject matter
    beginning_end = min(4, len(sentences))
    query1_sents = sentences[:beginning_end]
    query1 = ' '.join(query1_sents)
    queries.append(query1)
    logger.debug(f"   Query 1 length: {len(query1.split())} words")
    
    # ✅ Query 2: MIDDLE PARAGRAPH - Supporting evidence (3-4 sentences from middle)
    # This has specific details and examples
    mid_start = max(beginning_end, len(sentences) // 3)
    mid_end = min(mid_start + 4, len(sentences))
    query2_sents = sentences[mid_start:mid_end]
    query2 = ' '.join(query2_sents)
    queries.append(query2)
    logger.debug(f"   Query 2 length: {len(query2.split())} words")
    
    # ✅ Query 3: END PARAGRAPH - Conclusions and related concepts (last 3-4 sentences)
    # This captures implications and final thoughts
    end_start = max(mid_end, len(sentences) - 4)
    query3_sents = sentences[end_start:]
    query3 = ' '.join(query3_sents)
    queries.append(query3)
    logger.debug(f"   Query 3 length: {len(query3.split())} words")
    
    # ✅ Clean and validate queries
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

def find_semantic_matches(
    doc_text: str,
    source_text: str,
    threshold: float = 0.50
) -> List[Dict]:
    """
    Find semantically similar passages using MiniLM embeddings.
    Detects paraphrased content from LLMs.
    
    threshold: 0.65+ = high confidence, 0.50+ = catch paraphrasing, 0.35+ = catch weak matches
    """
    try:
        encoder = get_encoder()
    except ImportError:
        logger.warning("⚠️  SentenceTransformer not available, falling back to string matching")
        return _fallback_semantic_matches(doc_text, source_text, threshold)
    
    # Split into sentences
    doc_sentences = split_into_sentences(doc_text)
    source_sentences = split_into_sentences(source_text)
    
    if not doc_sentences or not source_sentences:
        return []
    
    logger.debug(f"   Encoding {len(doc_sentences)} doc sentences + {len(source_sentences)} source sentences...")
    
    # Encode all sentences
    try:
        doc_embeddings = encoder.encode(doc_sentences, convert_to_numpy=True, show_progress_bar=False)
        source_embeddings = encoder.encode(source_sentences, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        logger.error(f"   Encoding error: {e}, falling back to string matching")
        return _fallback_semantic_matches(doc_text, source_text, threshold)
    
    matches = []
    matched_source_indices = set()
    
    # Compare using cosine similarity
    for doc_idx, doc_emb in enumerate(doc_embeddings):
        best_similarity = 0.0
        best_source_idx = -1
        
        for source_idx, source_emb in enumerate(source_embeddings):
            if source_idx in matched_source_indices:
                continue
            
            # Cosine similarity
            similarity = np.dot(doc_emb, source_emb) / (
                np.linalg.norm(doc_emb) * np.linalg.norm(source_emb) + 1e-8
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_source_idx = source_idx
        
        # Record if above threshold
        if best_similarity >= threshold and best_source_idx >= 0:
            if best_source_idx not in matched_source_indices:
                matched_source_indices.add(best_source_idx)
                
                matches.append({
                    'doc_text': doc_sentences[doc_idx],
                    'source_text': source_sentences[best_source_idx],
                    'similarity': float(best_similarity),
                    'doc_index': doc_idx,
                    'source_index': best_source_idx
                })
    
    logger.debug(f"   Found {len(matches)} semantic matches (threshold: {threshold})")
    return matches

def _fallback_semantic_matches(doc_text: str, source_text: str, threshold: float) -> List[Dict]:
    """Fallback to string matching if embeddings not available"""
    from difflib import SequenceMatcher
    
    doc_sentences = split_into_sentences(doc_text)
    source_sentences = split_into_sentences(source_text)
    
    if not doc_sentences or not source_sentences:
        return []
    
    matches = []
    matched_source_indices = set()
    
    for doc_idx, doc_sent in enumerate(doc_sentences):
        best_similarity = 0.0
        best_source_idx = -1
        
        for source_idx, source_sent in enumerate(source_sentences):
            if source_idx in matched_source_indices:
                continue
            
            ratio = SequenceMatcher(None, doc_sent.lower(), source_sent.lower()).ratio()
            
            if ratio > best_similarity:
                best_similarity = ratio
                best_source_idx = source_idx
        
        if best_similarity >= threshold and best_source_idx >= 0:
            if best_source_idx not in matched_source_indices:
                matched_source_indices.add(best_source_idx)
                
                matches.append({
                    'doc_text': doc_sentences[doc_idx],
                    'source_text': source_sentences[best_source_idx],
                    'similarity': float(best_similarity),
                    'doc_index': doc_idx,
                    'source_index': best_source_idx
                })
    
    return matches

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts"""
    try:
        encoder = get_encoder()
        embeddings = encoder.encode([text1, text2], convert_to_numpy=True, show_progress_bar=False)
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
        )
        return float(similarity)
    except:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def compare_semantic_chunks(
    doc_text: str,
    source_text: str,
    chunk_size: int = 200,
    threshold: float = 0.65
) -> List[Dict]:
    """
    Compare document chunks against source chunks using semantic similarity.
    """
    def chunk_text(text, size):
        words = text.split()
        chunks = []
        for i in range(0, len(words), size):
            chunk = ' '.join(words[i:i+size])
            if len(chunk.split()) >= 20:
                chunks.append(chunk)
        return chunks
    
    doc_chunks = chunk_text(doc_text, chunk_size)
    source_chunks = chunk_text(source_text, chunk_size)
    
    if not doc_chunks or not source_chunks:
        return []
    
    try:
        encoder = get_encoder()
        doc_embeddings = encoder.encode(doc_chunks, convert_to_numpy=True, show_progress_bar=False)
        source_embeddings = encoder.encode(source_chunks, convert_to_numpy=True, show_progress_bar=False)
        
        matches = []
        for doc_emb in doc_embeddings:
            for source_emb in source_embeddings:
                similarity = np.dot(doc_emb, source_emb) / (
                    np.linalg.norm(doc_emb) * np.linalg.norm(source_emb) + 1e-8
                )
                
                if similarity >= threshold:
                    matches.append({
                        'doc_text': doc_chunks[0][:200] + "...",
                        'source_text': source_chunks[0][:200] + "...",
                        'similarity': float(similarity),
                    })
        
        return matches
    except:
        return []