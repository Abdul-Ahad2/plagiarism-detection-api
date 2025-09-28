
# Plagiarism Detection API (FastAPI)

A backend service built with FastAPI that performs Lexical, Semantic, and Internal plagiarism analysis on uploaded documents.
It is intended for students, teachers, researchers, and institutions to detect overlaps, possible plagiarism, or AI-assisted rewriting.

## Features

Lexical Analysis – Detects word/phrase overlaps, n-gram similarity.

Semantic Analysis – Embedding-based similarity search against an external MongoDB corpus.

Internal Analysis – Compares uploaded documents against each other.

LLM Narrative – Generates teacher-style plagiarism summaries.

JWT Authentication – Secure access to all routes.

## Installation

```git clone https://github.com/Abdul-Ahad2/plagiarism-detection-api.git```

```cd plagiarism-api```

```pip install -r requirements.txt```

Configuration

All environment values are stored in app/config.py.

```MONGODB_URI=your-mongodb-uri```

```MIN_WORDS_PER_SENTENCE = 4```

```MIN_SENTENCE_LENGTH = 30```

```SEQUENCE_THRESHOLD = 0.75```

```TFIDF_THRESHOLD = 0.80```

```SUB_PHRASE_TFIDF_MIN = 0.50```

```EXACT_MATCH_SCORE = 1.0```

```ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}```

```JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_nextauth_secret")```

```JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")```


Run Locally ```uvicorn app.main:app --reload```

## API Reference

Lexical Analysis (student)
```POST /student/lexical-analysis```.
Finds lexical overlap for a single file (phrases, n-grams).

Semantic Analysis (teacher)
```POST /teacher/semantic-analysis```. Uploads documents and compares them with external corpus.

Internal Analysis (teacher)
```POST /teacher/internal-analysis```.Cross-compares uploaded documents with each other.

Lexical Analysis (teacher)
```POST /teacher/lexical-analysis```.
Finds lexical overlap (phrases, n-grams).

Example Response (Semantic)
```{
  "id": "teacher_semantic",
  "name": "Teacher Semantic Analysis",
  "analysisType": "semantic",
  "mode": "external",
  "uploadDate": "2025-09-27T12:34:56Z",
  "processingTime": "0m 12s",
  "documents": [
    { "id": 1, "name": "H.docx" },
    { "id": 2, "name": "uzair.docx" }
  ],
  "summary": {
    "totalDocuments": 2,
    "flaggedComparisons": 2,
    "highestSimilarity": 97.3,
    "averageSimilarity": 95.2
  },
  "narrative": "The document 'H.docx' shows strong semantic overlap..."
}
```
