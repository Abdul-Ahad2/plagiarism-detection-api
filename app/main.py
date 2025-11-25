import logging
from fastapi import FastAPI
from app.routers.student.lexical_analysis import router as student_lexical_router
from app.routers.teacher.semantic_analysis import router as teacher_semantic_router
from app.routers.teacher.lexical_analysis import router as teacher_lexical_router
from app.routers.teacher.internal_analysis import router as teacher_internal_router

from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://plagiarism-detection-frontend.vercel.app", "*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(student_lexical_router)
app.include_router(teacher_internal_router)
app.include_router(teacher_lexical_router)
app.include_router(teacher_semantic_router)
