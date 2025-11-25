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
    contentA: str = ""
    contentB: str = ""

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
    content: Optional[str] = None 

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
    
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class CodeMatch(BaseModel):
    matched_code: str
    similarity: float
    source_type: str  # 'peer', 'github', 'stackoverflow', 'web'
    source_title: str
    source_url: Optional[str] = None
    match_type: str  # 'exact', 'structural', 'token_sequence'
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    context: Optional[str] = None

class CodeFunction(BaseModel):
    name: str
    start_line: int
    end_line: int
    code: str
    complexity: int  # Cyclomatic complexity
    tokens: List[str]
    ast_hash: str

class CodeDocResult(BaseModel):
    id: int
    name: str
    author: Optional[str] = None
    similarity: float
    flagged: bool
    lineCount: int
    functionCount: int
    matches: List[CodeMatch]
    functions: List[CodeFunction]
    content: str  # Full code content
    language: str

class CodeAnalysisSummary(BaseModel):
    totalDocuments: int
    flaggedDocuments: int
    highestSimilarity: float
    averageSimilarity: float
    totalMatches: int
    peerMatches: int
    externalMatches: int

class TeacherCodeBatchReport(BaseModel):
    id: str
    name: str
    uploadDate: datetime
    processingTime: str
    documents: List[CodeDocResult]
    summary: CodeAnalysisSummary
    assignmentTopic: Optional[str] = None

class InternalMatch(BaseModel):
    """Match between two student submissions"""
    student1_id: int
    student2_id: int
    student1_name: str
    student2_name: str
    similarity: float
    match_type: str
    matched_functions: List[str]
