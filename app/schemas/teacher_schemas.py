from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# ---- Shared ----

class DocumentInfo(BaseModel):
    id: int
    name: str
    author: Optional[str] = None

class OverlapDetail(BaseModel):
    # For lexical/internal
    fromDoc: str
    toDoc: str
    text: str
    similarity: float  # percent (0–100)
    sectionA: Optional[str] = None
    sectionB: Optional[str] = None
    context: Optional[str] = None

class ComparisonDetail(BaseModel):
    id: str         # "i-j"
    docA: str
    docB: str
    similarity: float  # percent (0–100)
    flagged: bool
    overlaps: List[OverlapDetail] = Field(default_factory=list)

class InternalReportSummary(BaseModel):
    totalDocuments: int
    totalComparisons: int
    flaggedComparisons: int
    highestSimilarity: float
    averageSimilarity: Optional[float] = None

class InternalReportDetail(BaseModel):
    id: str
    name: str
    analysisType: str = "internal"
    uploadDate: datetime
    processingTime: str
    status: str = "completed"
    documents: List[DocumentInfo]
    comparisons: List[ComparisonDetail]
    summary: InternalReportSummary



class LexicalMatch(BaseModel):
    matched_text: str
    similarity: float  
    source_type: str
    source_title: str
    source_url: str
    section: Optional[str] = None
    context: Optional[str] = None

class LexicalDocResult(BaseModel):
    id: int
    name: str
    author: Optional[str] = None
    similarity: float  # overall percent
    flagged: bool
    wordCount: Optional[int] = None
    matches: List[LexicalMatch] = Field(default_factory=list)

class TeacherLexicalSummary(BaseModel):
    totalDocuments: int
    flaggedDocuments: int
    highestSimilarity: float
    averageSimilarity: Optional[float] = None
    totalMatches: int

class TeacherLexicalBatchReport(BaseModel):
    id: str
    name: str
    analysisType: str = "lexical"
    uploadDate: datetime
    processingTime: str
    status: str = "completed"
    documents: List[LexicalDocResult]
    summary: TeacherLexicalSummary

# ---- Teacher Semantic (internal/external) ----

class SemanticOverlap(BaseModel):
    textA: str
    textB: str
    cosine: float     # 0–1
    cosine_pct: float # 0–100
    sectionA: Optional[str] = None
    sectionB: Optional[str] = None
    confidence: str   # "high" | "medium" | "low"

class SemanticComparison(BaseModel):
    id: str
    docA: str
    docB: str
    similarity: float   # aggregated cosine percent
    flagged: bool
    overlaps: List[SemanticOverlap] = Field(default_factory=list)

class TeacherSemanticReport(BaseModel):
    id: str
    name: str
    analysisType: str = "semantic"
    mode: str           # "internal" | "external"
    uploadDate: datetime
    processingTime: str
    status: str = "completed"
    documents: List[DocumentInfo]
    comparisons: List[SemanticComparison]
    summary: InternalReportSummary
    narrative: Optional[str] = None  
