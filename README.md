# Plagiarism Detection API

A comprehensive FastAPI-based plagiarism detection system that uses semantic analysis, web scraping, and AI detection to identify plagiarized content.

## Features

### Student Features
- **Single File Lexical Analysis** - Analyze one document for plagiarism
- **Exact Phrase Matching** - Detect copied text using TF-IDF
- **Web Source Comparison** - Find original sources
- **Fast Processing** - Quick turnaround on single files

### Teacher Features
- **Multiple File Upload** - Batch analyze multiple documents at once
- **Semantic Analysis** - Detect paraphrased and rephrased content
- **Lexical Analysis** - Find exact and near-exact matches
- **Internal Analysis** - Check for self-plagiarism and duplicate sections
- **AI Detection** - Identify AI-generated text
- **Comprehensive Reports** - Detailed similarity scores and matched passages
- **Batch Processing** - Process multiple files in one request
- **Advanced Filtering** - View high-confidence matches only

## 📋 Prerequisites

- Python 3.11+
- MongoDB connection string
- Google Custom Search API keys and Search Engine IDs
- Hugging Face token (optional, for inference API)

## Installation

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Abdul-Ahad2/plagiarism-detection-api
cd plagiarism-detection-api
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
playwright install chromium
python -m nltk.downloader stopwords wordnet punkt_tab
```

4. **Create `.env` file:**
```env
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/database
JWT_SECRET_KEY=your_jwt_secret
JWT_ALGORITHM=HS256
HF_TOKEN=your_huggingface_token
API_KEYS=key1,key2,key3
SEARCH_ENGINE_IDS=engine_id_1,engine_id_2
SECRET_KEY=your_secret_key
ALGORITHM=HS256
```

5. **Run the server:**
```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for API documentation.

## Deployment

### Deploy on Render

1. Push code to GitHub
2. Connect GitHub repo to Render
3. Set environment variables in Render dashboard
4. Deploy

Your API will be available at: `https://your-service-name.onrender.com`

### Deploy on Hugging Face Spaces

1. Create new Space (Docker runtime)
2. Upload project files and Dockerfile
3. Set secrets in Space settings
4. Auto-deploys on push

## API Endpoints

### 1. Semantic Analysis
**Endpoint:** `POST /teacher/semantic-analysis`

**Headers:**
```
Authorization: Bearer {jwt_token}
Content-Type: multipart/form-data
```

**Request:**
```json
{
  "files": [file1, file2, ...]
}
```

**Response:**
```json
{
  "id": "report_id",
  "name": "Semantic Analysis",
  "uploadDate": "2025-11-30T14:56:00",
  "processingTime": "5m 24s",
  "documents": [
    {
      "id": 1,
      "name": "document.pdf",
      "similarity": 85.3,
      "aiSimilarity": 0.765,
      "flagged": true,
      "wordCount": 258,
      "matches": [
        {
          "matched_text": "The text that was flagged as plagiarized...",
          "similarity": 85.1,
          "source_url": "https://example.com/article",
          "source_title": "Original Article",
          "source_type": "web"
        }
      ]
    }
  ],
  "summary": {
    "totalDocuments": 1,
    "flaggedDocuments": 1,
    "highestSimilarity": 85.3,
    "averageSimilarity": 85.3,
    "totalMatches": 9,
    "averageAiSimilarity": 0.765
  }
}
```

**Description:** Analyzes documents using semantic embeddings to detect paraphrased content from web sources. Returns similarity scores and matched passages.

---

### 2. Lexical Analysis
**Endpoint:** `POST /student/lexical-analysis`

**Headers:**
```
Authorization: Bearer {jwt_token}
Content-Type: multipart/form-data
```

**Request:**
```json
{
  "file": file
}
```

**Response:**
```json
{
  "id": "report_id",
  "name": "Lexical Analysis",
  "uploadDate": "2025-11-30T14:56:00",
  "processingTime": "2m 15s",
  "document": {
    "id": 1,
    "name": "essay.docx",
    "similarity": 72.4,
    "flagged": true,
    "wordCount": 1250,
    "matches": [
      {
        "matched_text": "These fragments show high lexical overlap with source material",
        "similarity": 89.2,
        "source_url": "https://example.com/paper",
        "source_title": "Academic Paper",
        "source_type": "web"
      }
    ]
  },
  "summary": {
    "highestSimilarity": 89.2,
    "averageSimilarity": 72.4,
    "totalMatches": 15,
    "detectionMethod": "Lexical matching with TF-IDF scoring"
  }
}
```

**Description:** Analyzes a single document for exact and near-exact lexical matches using TF-IDF and sequence matching. Identifies copied phrases and specific wording.

---

### 3. Internal Analysis
**Endpoint:** `POST /student/internal-analysis`

**Headers:**
```
Authorization: Bearer {jwt_token}
Content-Type: multipart/form-data
```

**Request:**
```json
{
  "files": [file1, file2, ...]
}
```

**Response:**
```json
{
  "id": "report_id",
  "name": "Internal Analysis",
  "uploadDate": "2025-11-30T14:56:00",
  "processingTime": "1m 45s",
  "documents": [
    {
      "id": 1,
      "name": "report.pdf",
      "similarity": 45.2,
      "flagged": false,
      "wordCount": 2500,
      "internalMatches": [
        {
          "section1": "Introduction",
          "section2": "Conclusion",
          "similarity": 67.8,
          "reason": "Duplicate content within same document"
        }
      ],
      "styleAnalysis": {
        "consistencyScore": 0.92,
        "readabilityGrade": 12,
        "diversityIndex": 0.78,
        "flaggedSections": []
      }
    }
  ],
  "summary": {
    "totalDocuments": 1,
    "flaggedDocuments": 0,
    "internalDuplicationRate": 5.3,
    "averageStyleConsistency": 0.92,
    "analysisType": "Internal plagiarism and self-plagiarism detection"
  }
}
```

**Description:** Detects internal plagiarism (copy-paste within same document/collection), self-plagiarism, and analyzes writing style consistency. Useful for identifying sections that don't match the author's typical style.

---

## How It Works

### 1. Semantic Analysis
- Extracts text from uploaded files
- Generates 5 semantic queries (beginning, early middle, center, late middle, end)
- Scrapes web for matching sources using Google Custom Search
- Compares document against sources using sentence-level embeddings
- Reports similarity scores and matched passages

### 2. AI Detection
Uses entropy analysis to detect AI-generated text:
- **Character Entropy** - Letter distribution patterns
- **Word Entropy** - Vocabulary diversity and structure
- **Word Length Entropy** - Consistency of word lengths
- Returns score 0-1 (1 = definitely AI)

### 3. Web Scraping
- Multi-threaded scraping with timeout protection
- Tries requests → cloudscraper → Playwright
- Respects politeness delays and blacklist domains
- Fallback to Google snippets if scraping fails

## Project Structure

```
plagiarism-detection-api/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Configuration & env variables
│   ├── routers/
│   │   ├── teacher/
│   │   │   └── semantic_analysis.py
│   │   └── ...
│   ├── schemas/                # Pydantic models
│   ├── utils/
│   │   ├── semantic_utils.py   # Semantic matching
│   │   ├── ai_detector.py      # AI detection
│   │   ├── web_utils.py        # Web scraping
│   │   └── file_utils.py       # File handling
│   └── logger.py               # Logging setup
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (local only)
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_URI` | MongoDB connection string | ✅ |
| `JWT_SECRET_KEY` | JWT signing secret | ✅ |
| `HF_TOKEN` | Hugging Face API token | ❌ |
| `API_KEYS` | Comma-separated Google API keys | ✅ |
| `SEARCH_ENGINE_IDS` | Comma-separated Search Engine IDs | ✅ |
| `SECRET_KEY` | General secret key | ✅ |
| `ALGORITHM` | JWT algorithm (HS256) | ✅ |

## Dependencies

- **fastapi** - Web framework
- **uvicorn** - ASGI server
- **motor** - Async MongoDB driver
- **pymongo** - MongoDB client
- **sentence-transformers** - Semantic embeddings
- **transformers** - AI detection models
- **torch** - Deep learning framework
- **playwright** - Browser automation
- **requests** - HTTP client
- **beautifulsoup4** - HTML parsing
- **pdfminer.six** - PDF text extraction
- **python-docx** - DOCX parsing
- **nltk** - Natural language toolkit

## Configuration

### Adjust Thresholds

Edit `config.py`:

```python
SEMANTIC_THRESHOLD = 0.50      # Semantic match threshold (0-1)
SCRAPING_TIMEOUT = 180         # Web scraping timeout (seconds)
REQUEST_TIMEOUT = 5            # Per-URL timeout (seconds)
MIN_TEXT_LENGTH = 700          # Minimum text for extraction
```

### Rate Limiting

```python
MAX_WORKERS = 8                # Parallel scraping threads
POLITENESS_DELAY = 0.1         # Delay between requests to same domain
```

## Troubleshooting

### "No sources found"
- Check internet connection
- Verify Google API keys are valid
- Check search engine ID is correct

### "Timeout exceeded"
- Increase `SCRAPING_TIMEOUT` in config
- Reduce number of parallel workers
- Check network stability

### "Model not found"
- Ensure `transformers` and `sentence-transformers` are installed
- Models auto-download on first use (~500MB)
- Check disk space

### MongoDB connection failed
- Verify `MONGODB_URI` is correct
- Check MongoDB cluster is online
- Ensure IP is whitelisted in MongoDB Atlas

## Performance Tips

1. **Use multiple API keys** - Distribute quota across keys
2. **Increase workers** - For faster scraping (but watch rate limits)
3. **Reduce timeout** - Fail faster on slow sites
4. **Cache models** - Pre-download models before deployment
5. **Use paid tier** - Render free tier spins down after inactivity

## Security

- JWT authentication required for endpoints
- Environment variables for all secrets
- `.env` excluded from git
- Rate limiting on API keys
- Input validation with Pydantic

## Support

For issues or questions:
- Open a GitHub issue
- Check existing documentation
- Review API logs in Render/Spaces dashboard
