import logging
from fastapi import FastAPI
from app.routers.student.lexical_analysis import router as plagiarism_router
from app.routers.teacher.semantic_analysis import router as semantic_router
from app.routers.teacher.lexical_analysis import router as lexical_router
from app.routers.teacher.internal_analysis import router as internal_router
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

app.include_router(plagiarism_router)
app.include_router(semantic_router)
app.include_router(lexical_router)
app.include_router(internal_router)