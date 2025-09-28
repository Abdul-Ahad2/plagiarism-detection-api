from pydantic import BaseModel
from typing import List

class MatchDetail(BaseModel):
    matched_text: str
    similarity: float
    source_type: str    # "news" or "academic"
    source_title: str
    source_url: str


class SentenceResult(BaseModel):
    original_sentence: str
    normalized_sentence: str
    match_type: str     # "full_sentence", "partial_phrase", "no_match"
    matches: List[MatchDetail]


class PlagiarismResponse(BaseModel):
    checked_sentences: int
    checked_sources: int
    results: List[SentenceResult]
