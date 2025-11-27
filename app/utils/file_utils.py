import tempfile
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document as DocxDocument
from app.config import ALLOWED_EXTENSIONS


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(content_bytes: bytes, filename: str, max_words: int = 500) -> str:
    ext = filename.rsplit(".", 1)[1].lower()
    text = ""
    try:
        if ext == "txt":
            text = content_bytes.decode("utf-8", errors="ignore")
        elif ext == "pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content_bytes)
                tmp.flush()
                text = extract_pdf_text(tmp.name)
        elif ext == "docx":
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(content_bytes)
                tmp.flush()
                doc = DocxDocument(tmp.name)
                text = "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        text = ""

    # Word count check
    word_count = len(text.split())
    if word_count > max_words:
        raise ValueError(f"File exceeds {max_words} words (found {word_count}).")

    return text
