import re
from typing import List, Optional, Tuple, Set
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rapidfuzz.distance import Levenshtein  # edit-distance similarity

from app.config import (
    MIN_WORDS_PER_SENTENCE,
    MIN_SENTENCE_LENGTH,
    SEQUENCE_THRESHOLD,   # use e.g. 0.75
    EXACT_MATCH_SCORE,    # e.g. 1.0
)

# Ensure required NLTK resources are downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

# ---------- Normalization ----------

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Basic lowercasing
    text = text.lower()

    # --- NEW: pre-clean artifacts that break shingles ---
    text = text.replace("\u00ad", "")                    # soft hyphen
    text = re.sub(r"-\s*\n\s*", "", text)                # hyphenated line breaks
    text = text.replace("\n", " ")
    # text = text.replace("&", " and ")                    # '&' -> 'and'
    text = re.sub(r"[^\x20-\x7E]+", " ", text)           # strip non-printables

    # Smart quotes / dashes
    text = (text.replace("’", "'").replace("‘", "'")
                 .replace("“", '"').replace("”", '"')
                 .replace("—", "-").replace("–", "-"))

    # Remove residual punctuation (keep hyphen)
    text = re.sub(r"[^\w\s-]", " ", text)

    # Collapse spaces *before* tokenization to avoid empty tokens
    text = re.sub(r"\s+", " ", text).strip()

    # Lemmatize
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
    normalized = " ".join(lemmas)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# ---------- Sentence filtering ----------

def get_meaningful_sentences(text: str) -> List[str]:
    sentences = sent_tokenize(text or "")
    filtered = []
    for s in sentences:
        words = word_tokenize(s)
        if len(words) >= MIN_WORDS_PER_SENTENCE and len(s.strip()) >= MIN_SENTENCE_LENGTH:
            filtered.append(s.strip())
    return filtered

# ---------- Keyword extraction (simple & fine) ----------

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    words = word_tokenize((text or "").lower())
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 3]
    freq = nltk.FreqDist(filtered)
    return [word for word, _ in freq.most_common(max_keywords)]

# ---------- Shingling & set-based overlap ----------

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

# ---------- Winnowing (fingerprinting) ----------

def _winnowing_hashes(norm_text: str, k: int = 7, w: int = 4) -> List[Tuple[int, int]]:
    """
    Returns fingerprints as (hash, shingle_start_idx_in_tokens).
    Min guaranteed match length ≈ k + w - 1 words.
    """
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
        # rightmost minimum
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

# ---------- Exact matching (substring), Edit distance, LCS ----------

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

# ---------- PUBLIC API (same names you already use) ----------

def find_exact_matches(sentence: str, external_text: str) -> Optional[float]:
    """
    Full-sentence lexical match (with LCS safety net):
      1) exact substring on normalized forms → EXACT_MATCH_SCORE
      2) winnowing fingerprint overlap → score in [0,1]
      3) edit-distance similarity → fallback
      4) LCS (longest common substring) length fallback (chars) → lcs_len/len(norm_s)
    """
    norm_s = normalize_text(sentence)
    norm_e = normalize_text(external_text)

    if len(norm_s) < MIN_SENTENCE_LENGTH:
        return None

    # 1) Exact full-sentence match
    if _exact_substring(norm_s, norm_e):
        return EXACT_MATCH_SCORE

    # 2) Winnowing overlap (robust exact-ish)
    win_sim = _winnowing_overlap(
        _winnowing_hashes(norm_s, k=5, w=4),
        _winnowing_hashes(norm_e, k=5, w=4),
    )
    if win_sim >= SEQUENCE_THRESHOLD:
        return round(win_sim, 3)

    # 3) Edit-distance fallback (paraphrase-lite)
    lev = _levenshtein_sim(norm_s, norm_e)
    if lev >= SEQUENCE_THRESHOLD:
        return round(lev, 3)

    # 4) LCS fallback (catches long contiguous overlaps missed above)
    #    Adjust threshold to your tolerance; 25–35 chars works well for academic text.
    lcs_len = _lcs_length(norm_s, norm_e)
    LCS_MIN_CHARS = 25  # <- tune: 25–35 typical
    if lcs_len >= LCS_MIN_CHARS:
        # Normalize by sentence length to keep output in [0,1]
        return round(lcs_len / max(1, len(norm_s)), 3)

    return None


def find_partial_phrase_match(sentence: str, external_text: str) -> Optional[Tuple[str, float]]:
    """
    Partial reuse via 7-word shingles:
      - compute Jaccard and Containment
      - if strong enough, return (representative_phrase, score)
    """
    norm_s = normalize_text(sentence)
    norm_e = normalize_text(external_text)
    A, B = _shingle_sets(norm_s, norm_e, k=7)
    if not A or not B:
        return None

    jac = _jaccard(A, B)
    con = _containment(A, B)
    score = max(jac, con)
    if score < SEQUENCE_THRESHOLD:
        return None

    # Representative phrase: first shared shingle mapped to original sentence (best-effort)
    shared = list(A & B)
    if shared:
        # return original sentence as the matched_text (simpler for UI)
        return sentence, round(score, 3)
    return None