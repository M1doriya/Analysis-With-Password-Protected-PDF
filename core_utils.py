"""core_utils.py

Project-wide utilities used by Streamlit apps and bank parsers.

Goals:
1) Standardize input handling (PDF bytes)
2) Standardize transaction schema and types
3) Make date/amount parsing resilient across banks
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# PDF INPUT
# -----------------------------
def read_pdf_bytes(pdf_input: Any) -> bytes:
    """Return PDF bytes from:
    - bytes / bytearray
    - Streamlit UploadedFile (has getvalue)
    - file-like objects (has read)
    - filesystem path (str)
    """
    if isinstance(pdf_input, (bytes, bytearray)):
        return bytes(pdf_input)

    # Streamlit UploadedFile
    if hasattr(pdf_input, "getvalue"):
        data = pdf_input.getvalue()
        if data:
            return data

    # file-like
    if hasattr(pdf_input, "read"):
        try:
            pdf_input.seek(0)
        except Exception:
            pass
        data = pdf_input.read()
        if data:
            return data

    # path
    if isinstance(pdf_input, str):
        with open(pdf_input, "rb") as f:
            return f.read()

    raise ValueError("Unable to read PDF bytes from the provided input")


def bytes_to_pdfplumber(pdf_bytes: bytes):
    """Helper to open pdfplumber using bytes."""
    import pdfplumber  # local import to keep utils lightweight
    return pdfplumber.open(BytesIO(pdf_bytes))


# -----------------------------
# NORMALIZATION
# -----------------------------
_WS_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    return _WS_RE.sub(" ", str(text or "")).strip()


def safe_float(value: Any) -> float:
    """Convert numeric strings to float safely.

    Handles:
    - None / empty
    - commas
    - parentheses negatives: (1,234.56)
    - trailing +/-: 123.45- / 123.45+
    - currency symbols and stray text
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return 0.0

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # trailing sign
    trailing_sign = None
    if s.endswith("+"):
        trailing_sign = "+"
        s = s[:-1].strip()
    elif s.endswith("-"):
        trailing_sign = "-"
        s = s[:-1].strip()

    s = s.replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", "."}:
        return 0.0

    try:
        f = float(s)
    except Exception:
        return 0.0

    if neg or trailing_sign == "-":
        f = -abs(f)
    elif trailing_sign == "+":
        f = abs(f)
    return float(f)


def normalize_date(date_value: Any, default_year: Optional[int] = None) -> Optional[str]:
    """Normalize many common bank-statement date formats to ISO YYYY-MM-DD.
    Returns None if parsing fails.
    """
    if date_value is None:
        return None

    s = normalize_text(date_value)
    if not s:
        return None

    # already ISO
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s

    # common patterns (day-first)
    patterns: List[Tuple[str, str]] = [
        (r"^\d{1,2}/\d{1,2}/\d{4}$", "%d/%m/%Y"),
        (r"^\d{1,2}-\d{1,2}-\d{4}$", "%d-%m-%Y"),
        (r"^\d{1,2}/\d{1,2}/\d{2}$", "%d/%m/%y"),
        (r"^\d{1,2}-\d{1,2}-\d{2}$", "%d-%m-%y"),
        (r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$", "%d %b %Y"),
        (r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{2}$", "%d %b %y"),
        (r"^\d{1,2}\s+[A-Za-z]{3}$", "%d %b"),
        (r"^\d{1,2}/\d{1,2}$", "%d/%m"),
        (r"^\d{1,2}-\d{1,2}$", "%d-%m"),
    ]

    for rx, fmt in patterns:
        if not re.fullmatch(rx, s):
            continue
        try:
            if fmt in {"%d %b", "%d/%m", "%d-%m"}:
                if default_year is None:
                    return None
                dt = datetime.strptime(f"{s} {default_year}", fmt + " %Y")
            else:
                dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # last-resort: dateutil
    try:
        from dateutil import parser as dateparser

        dt = dateparser.parse(
            s,
            dayfirst=True,
            yearfirst=False,
            default=datetime(default_year or 2000, 1, 1),
        )
        # if no explicit year and default_year is None, dt will use 2000 and likely be wrong -> reject
        if default_year is None and dt.year == 2000 and not re.search(r"\b\d{4}\b", s):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def infer_default_year(transactions: Iterable[Dict[str, Any]]) -> Optional[int]:
    """Infer a reasonable default year from any transaction that already contains a year."""
    for tx in transactions:
        d = normalize_text(tx.get("date"))
        if re.search(r"\b\d{4}\b", d):
            iso = normalize_date(d)
            if iso:
                return int(iso[:4])
    return None


def ensure_transaction_schema(
    tx: Dict[str, Any],
    *,
    default_bank: str,
    default_source_file: str,
    default_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a sanitized transaction dict with consistent keys and types."""
    raw_date = tx.get("date")
    date_iso = normalize_date(raw_date, default_year=default_year)

    description = normalize_text(tx.get("description"))
    debit = safe_float(tx.get("debit", 0))
    credit = safe_float(tx.get("credit", 0))

    # Some parsers store negative values; normalize to non-negative debit/credit where possible
    if debit < 0 and credit == 0:
        credit = abs(debit)
        debit = 0.0
    if credit < 0 and debit == 0:
        debit = abs(credit)
        credit = 0.0

    balance_raw = tx.get("balance", None)
    balance = safe_float(balance_raw) if balance_raw is not None and str(balance_raw).strip() != "" else None

    page_raw = tx.get("page")
    try:
        page = int(page_raw) if page_raw is not None and str(page_raw).strip() != "" else None
    except Exception:
        page = None

    bank = normalize_text(tx.get("bank")) or default_bank
    source_file = normalize_text(tx.get("source_file")) or default_source_file

    out: Dict[str, Any] = {
        "date": date_iso or normalize_text(raw_date),
        "description": description,
        "debit": round(float(debit), 2),
        "credit": round(float(credit), 2),
        "balance": round(float(balance), 2) if isinstance(balance, (int, float)) else None,
        "page": page,
        "bank": bank,
        "source_file": source_file,
    }

    # retain raw date if normalization changed it
    if date_iso and normalize_text(raw_date) and normalize_text(raw_date) != date_iso:
        out["_raw_date"] = normalize_text(raw_date)

    return out


def normalize_transactions(
    transactions: List[Dict[str, Any]],
    *,
    default_bank: str,
    source_file: str,
) -> List[Dict[str, Any]]:
    """Normalize a list of transactions and infer year if needed."""
    year = infer_default_year(transactions)
    return [
        ensure_transaction_schema(
            tx,
            default_bank=default_bank,
            default_source_file=source_file,
            default_year=year,
        )
        for tx in transactions
    ]


def transaction_fingerprint(tx: Dict[str, Any]) -> str:
    """Create a stable fingerprint suitable for de-duplication."""
    parts = [
        normalize_text(tx.get("date")),
        normalize_text(tx.get("description")),
        f"{safe_float(tx.get('debit', 0)):.2f}",
        f"{safe_float(tx.get('credit', 0)):.2f}",
        "" if tx.get("balance") is None else f"{safe_float(tx.get('balance')):.2f}",
        normalize_text(tx.get("bank")),
    ]
    blob = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(blob).hexdigest()


def dedupe_transactions(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for tx in transactions:
        fp = transaction_fingerprint(tx)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(tx)
    return out


# =========================================================
# Affin-specific fixes (DO NOT affect other banks unless called)
# =========================================================
def dedupe_transactions_affin(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Affin statements are frequently OCR-based; description strings vary across files.
    De-dupe must NOT depend on description/page/source_file, or overlap PDFs will inflate totals.

    Key:
      (date, debit, credit, balance, bank)
    """
    seen = set()
    out = []
    for tx in transactions:
        date = normalize_text(tx.get("date"))
        bank = normalize_text(tx.get("bank"))
        debit = round(safe_float(tx.get("debit", 0.0)), 2)
        credit = round(safe_float(tx.get("credit", 0.0)), 2)
        bal_raw = tx.get("balance")
        balance = None if bal_raw is None else round(safe_float(bal_raw), 2)

        key = (date, debit, credit, balance, bank)
        if key in seen:
            continue
        seen.add(key)
        out.append(tx)
    return out


def filter_affin_balance_outliers(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop rows whose balance is a clear OCR outlier (e.g. extra digit -> millions),
    which causes massive delta-based phantom debits/credits.

    Method:
      - compute median balance
      - keep balances within +/- 1.5M of median
      - keep rows with balance=None unchanged
    """
    bals = [safe_float(t.get("balance")) for t in transactions if t.get("balance") is not None]
    if len(bals) < 10:
        return transactions

    bals_sorted = sorted(bals)
    median = bals_sorted[len(bals_sorted) // 2]

    lo = median - 1_500_000
    hi = median + 1_500_000

    out = []
    for t in transactions:
        b = t.get("balance")
        if b is None:
            out.append(t)
            continue
        bf = safe_float(b)
        if lo <= bf <= hi:
            out.append(t)

    return out
