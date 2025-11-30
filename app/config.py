import os
from dotenv import load_dotenv

load_dotenv()

# ───── API Keys & URLs ─────
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://localhost:27017/sluethink")

# ───── Similarity thresholds ─────
MIN_WORDS_PER_SENTENCE = 4
MIN_SENTENCE_LENGTH = 30
SEQUENCE_THRESHOLD = 0.75
TFIDF_THRESHOLD = 0.80
SUB_PHRASE_TFIDF_MIN = 0.50
EXACT_MATCH_SCORE = 1.0

# ───── File support ─────
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_nextauth_secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
HF_TOKEN = os.getenv("HF_TOKEN", "")

API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
SEARCH_ENGINE_IDS = os.getenv("SEARCH_ENGINE_IDS", "").split(",") if os.getenv("SEARCH_ENGINE_IDS") else []

SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ALGORITHM = os.getenv("ALGORITHM", "HS256")

MIN_FUNCTION_LINES = 5
MIN_CODE_BLOCK_LINES = 3
STRUCTURAL_SIMILARITY_THRESHOLD = 0.75
TOKEN_SIMILARITY_THRESHOLD = 0.70
EXACT_MATCH_THRESHOLD = 0.90

MAX_QUERIES_PER_SUBMISSION = 3
RESULTS_PER_QUERY = 10
MAX_SOURCES_TO_ANALYZE = 8

REQUEST_TIMEOUT = 6
MAX_SCRAPE_WORKERS = 4
POLITENESS_DELAY = 0.2

ALLOWED_CODE_EXTENSIONS = {'.py', '.java', '.cpp', '.c', '.js', '.jsx', '.ts', '.tsx', '.cs', '.rb', '.go', '.php'}
MAX_FILE_SIZE_MB = 5

LOG_LEVEL = "INFO"
