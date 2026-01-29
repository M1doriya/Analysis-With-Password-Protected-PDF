"""
ocr_fallback.py

Mandatory OCR fallback for scanned (image-only) PDFs when the normal parser returns no transactions.

The goal is "best effort" extraction: we try to detect transaction rows by date + amounts + balance.
This is intentionally bank-agnostic so it can be applied to *any* bank when needed.

Dependencies:
- PyMuPDF (fitz)
- Pillow
- pytesseract
- tesseract-ocr system package (via packages.txt on Streamlit Cloud)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from core_utils import normalize_date, normalize_text, safe_float


_MONEY_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*\.\d{2}|\-?\d+\.\d{2}")

# Date-like token at start of a line.
_DATE_START_RE = re.compile(
    r"^\s*(?P<date>(\d{2}[/-]\d{2}[/-]\d{2,4})|(\d{6})|(\d{8})|(\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4}))\b"
)

_OPENING_RE = re.compile(
    r"\b(OPENING\s+BALANCE|BEGINNING\s+BALANCE|BALANCE\s+B/?F|BALANCE\s+BROUGHT\s+FORWARD)\b",
    re.I,
)
_ENDING_RE = re.compile(r"\b(ENDING\s+BALANCE|CLOSING\s+BALANCE)\b", re.I)


def ocr_pdf_to_text_pages(pdf_bytes: bytes, dpi: int = 220, max_pages: Optional[int] = None) -> List[str]:
    """
    Render each page to an image and run Tesseract OCR.
    Returns one OCR text blob per page.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text: List[str] = []

    try:
        for i, page in enumerate(doc):
            if max_pages is not None and i >= max_pages:
                break

            # Scale from 72dpi "points" to requested dpi
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            text = pytesseract.image_to_string(img, lang="eng") or ""
            pages_text.append(text)
    finally:
        doc.close()

    return pages_text


def _extract_money_values(line: str) -> List[float]:
    vals = []
    for m in _MONEY_RE.findall(line or ""):
        v = safe_float(m)
        if v is not None:
            vals.append(v)
    return vals


def _balance_sign(line: str) -> int:
    # Many statements annotate balance with CR/DR
    tail = (line or "").strip().split()[-1].upper() if (line or "").strip().split() else ""
    return -1 if tail == "DR" else 1


@dataclass
class _OcrTxn:
    date_iso: str
    desc: str
    amount: Optional[float]
    balance: Optional[float]


def parse_transactions_from_ocr_text_pages(pages_text: List[str], bank_name: str, source_file: str) -> List[dict]:
    """
    Parse OCR text into approximate transactions.

    Strategy:
      - line must start with a date token
      - line must contain at least 1 money token
      - last money token is treated as balance
      - second last (if any) treated as amount
      - debit/credit inferred from balance delta when possible, else inferred from CR/DR.

    Returns standard schema rows.
    """
    txns: List[dict] = []

    prev_balance: Optional[float] = None

    for page_idx, text in enumerate(pages_text or [], start=1):
        lines = [normalize_text(x) for x in (text or "").splitlines() if normalize_text(x)]

        for line in lines:
            up = line.upper()

            if _OPENING_RE.search(up) or _ENDING_RE.search(up):
                # Keep balance markers as non-real txns if money exists
                vals = _extract_money_values(line)
                if vals:
                    bal = float(vals[-1]) * _balance_sign(line)
                    marker = "OPENING BALANCE" if _OPENING_RE.search(up) else "ENDING BALANCE"
                    txns.append(
                        {
                            "date": None,
                            "description": marker,
                            "debit": 0.0,
                            "credit": 0.0,
                            "balance": round(bal, 2),
                            "page": page_idx,
                            "bank": bank_name,
                            "source_file": source_file,
                            "is_balance_marker": True,
                            "format": "ocr_marker",
                        }
                    )
                    prev_balance = bal
                continue

            m = _DATE_START_RE.match(line)
            if not m:
                continue

            date_token = m.group("date")
            date_iso = normalize_date(date_token)
            if not date_iso:
                continue

            vals = _extract_money_values(line)
            if not vals:
                continue

            bal = float(vals[-1]) * _balance_sign(line)
            amt = float(vals[-2]) if len(vals) >= 2 else None

            # Build description: remove leading date token and money tokens
            desc = line[len(date_token) :].strip()
            for mv in _MONEY_RE.findall(desc):
                desc = desc.replace(mv, " ")
            desc = normalize_text(desc)

            debit = 0.0
            credit = 0.0

            if prev_balance is not None and bal is not None:
                delta = round(bal - prev_balance, 2)
                if delta > 0:
                    credit = abs(delta)
                elif delta < 0:
                    debit = abs(delta)
            else:
                # fallback based on CR/DR and amount
                if amt is not None:
                    if _balance_sign(line) < 0:
                        # DR tends to reflect negative balance, not txn direction; so we don't force it.
                        pass
                    # naive: if keywords indicate inbound
                    if any(k in up for k in ["CR", "CREDIT", "DEPOSIT", "DUITNOW CR"]):
                        credit = abs(amt)
                    else:
                        debit = abs(amt)

            prev_balance = bal

            txns.append(
                {
                    "date": date_iso,
                    "description": desc,
                    "debit": round(float(debit), 2),
                    "credit": round(float(credit), 2),
                    "balance": round(float(bal), 2) if bal is not None else None,
                    "page": page_idx,
                    "bank": bank_name,
                    "source_file": source_file,
                    "format": "ocr_best_effort",
                }
            )

    return txns
