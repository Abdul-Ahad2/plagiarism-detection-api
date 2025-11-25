import re
from typing import List, Optional, Tuple, Set
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rapidfuzz.distance import Levenshtein

from app.config import (
    MIN_WORDS_PER_SENTENCE,
    MIN_SENTENCE_LENGTH,
    SEQUENCE_THRESHOLD,
    EXACT_MATCH_SCORE,
)

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"[^\x20-\x7E]+", " ", text)
    text = (text.replace("'", "'").replace("'", "'")
                 .replace(""", '"').replace(""", '"')
                 .replace("—", "-").replace("–", "-"))
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
    normalized = " ".join(lemmas)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def get_meaningful_sentences(text: str) -> List[str]:
    """Extract meaningful sentences from text."""
    sentences = sent_tokenize(text or "")
    filtered = []
    for s in sentences:
        words = word_tokenize(s)
        if len(words) >= MIN_WORDS_PER_SENTENCE and len(s.strip()) >= MIN_SENTENCE_LENGTH:
            filtered.append(s.strip())
    return filtered

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract top keywords from text for search queries."""
    words = word_tokenize((text or "").lower())
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 3]
    freq = nltk.FreqDist(filtered)
    return [word for word, _ in freq.most_common(max_keywords)]

def _word_shingles(norm_text: str, k: int = 7) -> List[str]:
    tokens = norm_text.split()
    if len(tokens) < k:
        return []
    return [" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]

def _shingle_sets(a_norm: str, b_norm: str, k: int = 7) -> Tuple[Set[str], Set[str]]:
    return set(_word_shingles(a_norm, k)), set(_word_shingles(b_norm, k))

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union

def _containment(a: Set[str], b: Set[str]) -> float:
    if not a:
        return 0.0
    inter = len(a & b)
    return inter / len(a)

def _winnowing_hashes(norm_text: str, k: int = 7, w: int = 4) -> List[Tuple[int, int]]:
    tokens = norm_text.split()
    if len(tokens) < k:
        return []
    shingles = [" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]
    hashes = [(hash(s) & 0xFFFFFFFF, i) for i, s in enumerate(shingles)]

    if w <= 1 or len(hashes) <= w:
        return list(dict.fromkeys(hashes))

    fps: List[Tuple[int, int]] = []
    last_min_abs = -1
    for i in range(0, len(hashes) - w + 1):
        window = hashes[i:i+w]
        min_hash, min_idx = None, None
        for j, (h, _) in enumerate(window):
            if (min_hash is None) or (h < min_hash) or (h == min_hash and j > (min_idx or -1)):
                min_hash, min_idx = h, j
        abs_idx = i + (min_idx or 0)
        if abs_idx != last_min_abs:
            fps.append(hashes[abs_idx])
            last_min_abs = abs_idx
    return list(dict.fromkeys(fps))

def _winnowing_overlap(a_fp: List[Tuple[int, int]], b_fp: List[Tuple[int, int]]) -> float:
    a_set, b_set = set(a_fp), set(b_fp)
    if not a_set or not b_set:
        return 0.0
    shared = len(a_set & b_set)
    denom = min(len(a_set), len(b_set)) or 1
    return shared / denom

def _exact_substring(norm_sentence: str, norm_external: str) -> bool:
    if not norm_sentence or not norm_external:
        return False
    return norm_external.find(norm_sentence) != -1

def _levenshtein_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return Levenshtein.normalized_similarity(a, b)

def _lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        curr = [0]
        ai = a[i-1]
        for j in range(1, len(b) + 1):
            if ai == b[j-1]:
                v = prev[j-1] + 1
                curr.append(v)
                if v > best:
                    best = v
            else:
                curr.append(0)
        prev = curr
    return best

def _extract_phrases(norm_text: str, min_words: int = 3, max_words: int = 7) -> List[str]:
    """Extract all phrases of varying lengths for partial match detection."""
    tokens = norm_text.split()
    phrases = []
    for k in range(min_words, min(max_words + 1, len(tokens) + 1)):
        for i in range(len(tokens) - k + 1):
            phrases.append(" ".join(tokens[i:i+k]))
    return phrases

def _phrase_containment_match(norm_sentence: str, norm_external: str) -> Optional[Tuple[str, float]]:
    """Check if ANY phrase (3-7 words) from sentence appears in external text."""
    phrases = _extract_phrases(norm_sentence, min_words=3, max_words=7)
    
    best_phrase = None
    best_score = 0.0
    
    for phrase in phrases:
        if phrase in norm_external:
            score = len(phrase.split()) / len(norm_sentence.split())
            if score > best_score:
                best_score = score
                best_phrase = phrase
    
    if best_phrase and best_score >= 0.4:
        return best_phrase, round(best_score, 3)
    
    return None

def find_exact_matches(sentence: str, external_text: str) -> Optional[float]:
    """
    Full-sentence lexical match with multiple strategies:
      1) Exact substring match → EXACT_MATCH_SCORE (100%)
      2) Winnowing fingerprints → robust plagiarism detection
      3) Edit-distance (Levenshtein) → catches paraphrased content
      4) LCS (longest common substring) → catches missed overlaps
    """
    norm_s = normalize_text(sentence)
    norm_e = normalize_text(external_text)

    if len(norm_s) < MIN_SENTENCE_LENGTH:
        return None

    # 1) Exact full-sentence match
    if _exact_substring(norm_s, norm_e):
        return EXACT_MATCH_SCORE

    # 2) Winnowing overlap
    win_sim = _winnowing_overlap(
        _winnowing_hashes(norm_s, k=5, w=4),
        _winnowing_hashes(norm_e, k=5, w=4),
    )
    if win_sim >= SEQUENCE_THRESHOLD * 0.9:
        return round(win_sim, 3)

    # 3) Edit-distance fallback
    lev = _levenshtein_sim(norm_s, norm_e)
    if lev >= SEQUENCE_THRESHOLD:
        return round(lev, 3)

    # 4) LCS fallback
    lcs_len = _lcs_length(norm_s, norm_e)
    LCS_MIN_CHARS = 15
    if lcs_len >= LCS_MIN_CHARS:
        return round(lcs_len / max(1, len(norm_s)), 3)

    return None


def find_partial_phrase_match(sentence: str, external_text: str) -> Optional[Tuple[str, float]]:
    """
    Partial reuse detection via multiple strategies:
      1) Phrase containment (3-7 word phrases)
      2) Shingle-based Jaccard/Containment
      3) Edit distance as fallback
    """
    norm_s = normalize_text(sentence)
    norm_e = normalize_text(external_text)
    
    if not norm_s or not norm_e:
        return None

    # Strategy 1: Check if any 3-7 word phrase appears verbatim
    phrase_match = _phrase_containment_match(norm_s, norm_e)
    if phrase_match:
        return sentence, phrase_match[1]

    # Strategy 2: 7-word shingles
    A, B = _shingle_sets(norm_s, norm_e, k=7)
    if A and B:
        jac = _jaccard(A, B)
        con = _containment(A, B)
        score = max(jac, con)
        if score >= SEQUENCE_THRESHOLD * 0.8:
            return sentence, round(score, 3)

    # Strategy 3: Smaller shingles (5-word) for more matches
    A_small, B_small = _shingle_sets(norm_s, norm_e, k=5)
    if A_small and B_small:
        jac_small = _jaccard(A_small, B_small)
        con_small = _containment(A_small, B_small)
        score_small = max(jac_small, con_small)
        if score_small >= SEQUENCE_THRESHOLD * 0.75:
            return sentence, round(score_small, 3)

    # Strategy 4: Edit distance as last resort
    lev = _levenshtein_sim(norm_s, norm_e)
    if lev >= SEQUENCE_THRESHOLD * 0.85:
        return sentence, round(lev, 3)

    return None

def find_partial_phrase_match_for_internal(sentence: str, external_text: str) -> Optional[Tuple[str, float]]:
    """
    Wrapper around find_partial_phrase_match that extracts the actual matched phrase
    that appears in BOTH documents, not just the full sentence from the first doc.
    """
    result = find_partial_phrase_match(sentence, external_text)
    if not result:
        return None
    
    matched_text, score = result
    norm_s = normalize_text(sentence)
    norm_e = normalize_text(external_text)
    
    if not norm_s or not norm_e:
        return result
    
    # Find the longest common substring that appears in both
    words_s = norm_s.split()
    words_e = norm_e.split()
    
    best_common = ""
    best_len = 0
    
    # Try all consecutive word sequences from sentence A
    for i in range(len(words_s)):
        for j in range(i + 1, len(words_s) + 1):
            phrase = ' '.join(words_s[i:j])
            # Check if this phrase exists in document B
            if phrase in norm_e:
                phrase_len = len(phrase)
                if phrase_len > best_len:
                    best_common = phrase
                    best_len = phrase_len
    
    # If we found a common phrase, return it
    if best_common:
        return best_common, score
    
    # Fallback: return original result
    return matched_text, score