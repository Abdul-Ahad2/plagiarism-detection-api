from pydantic import BaseModel
from typing import List
from datetime import datetime
from app.schemas.plagiarism_schemas import MatchDetail

class ReportSummary(BaseModel):
    id: str                # MongoDB’s ObjectId as string
    name: str
    date: datetime
    similarity: float       # highest sentence‐level similarity (0–100)
    sources: List[str]      # unique list of source titles used
    word_count: int
    time_spent: str         # e.g. "00:00"
    flagged: bool           # true if similarity > 70%


class ReportDetail(BaseModel):
    id: str
    name: str
    content: str
    plagiarism_data: List[MatchDetail]