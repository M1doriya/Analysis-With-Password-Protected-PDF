"""
pdf_security.py

Utilities for handling password-protected (encrypted) PDFs.

We decrypt into a new in-memory PDF (bytes) so downstream libraries like pdfplumber and PyMuPDF
can read it without needing to pass passwords around.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

from pypdf import PdfReader, PdfWriter


def is_pdf_encrypted(pdf_bytes: bytes) -> bool:
    """
    Returns True if the PDF is encrypted (password-protected).
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        return bool(reader.is_encrypted)
    except Exception:
        # If parsing fails, treat as "possibly encrypted" so UI can request a password.
        return True


def decrypt_pdf_bytes(pdf_bytes: bytes, password: Optional[str]) -> bytes:
    """
    Decrypt an encrypted PDF using the provided password and return a decrypted PDF as bytes.

    Raises ValueError if:
      - PDF is encrypted and password is missing
      - password is incorrect
      - PDF cannot be decrypted
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    if not reader.is_encrypted:
        return pdf_bytes

    if not password:
        raise ValueError("Password required for encrypted PDF.")

    result = reader.decrypt(password)
    # pypdf returns 0 on failure; 1/2 on success.
    if result == 0:
        raise ValueError("Incorrect password or unable to decrypt PDF.")

    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    out = BytesIO()
    writer.write(out)
    return out.getvalue()
