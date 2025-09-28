import os

# ───── API Keys & URLs ─────

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://me:123@cluster0.kjvqwtb.mongodb.net/sluethink?retryWrites=true&w=majority&appName=Cluster0")

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