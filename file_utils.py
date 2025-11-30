# file_utils.py
from typing import Optional
from io import BytesIO

from PyPDF2 import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> Optional[str]:
    """
    Extract text from a PDF (bytes).
    Returns None if extraction fails or is empty.
    """
    try:
        reader = PdfReader(BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            texts.append(txt)
        full_text = "\n".join(texts).strip()
        return full_text or None
    except Exception:
        return None
