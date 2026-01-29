"""
core_utils.py

Project-wide utilities used by Streamlit apps and bank parsers.

Design goals:
- Provide a stable, backward-compatible API for all bank parsers
- Standardize transaction schema across banks
- Robust date + money parsing
- Safe dedupe helpers
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# PDF INPUT
# =============================================================================
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
        data = pdf_input.read()
        if data:
            return data

    # filesystem path
    if isinstance(pdf_input, str):
        with open(pdf_input, "rb") as f:
            return f.read()

    raise TypeError(f"Unsupported PDF input type: {type(pdf_input)}")


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================
_WS_RE = re.compile(r"\s+")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]+")


def normalize_text(x: Any) -> str:
    """Normalize to a clean single-line string."""
    if x is None:
        return ""
    s = str(x)
    s = _NON_PRINTABLE_RE.sub(" ", s)
    s = s.replace("\u00a0", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s


# =============================================================================
# MONEY / NUMBER PARSING
# =============================================================================
_MONEY_TOKEN_RE = re.compile(r"^-?\(?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\)?[+-]?$|^-?\d+(?:\.\d{2})?[+-]?$")


def safe_float(x: Any, default: float = 0.0) -> float:
    """
    Parse common money-like strings into float.
    Supports:
      "1,234.56"
      "-1,234.56"
      "(1,234.56)"  -> -1234.56
      "1,234.56-"   -> -1234.56
      "1,234.56+"   -> +1234.56
      "RM 1,234.56"
    """
    if x is None:
        return default
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except Exception:
            return default

    s = normalize_text(x)
    if not s:
        return default

    # strip currency
    s = s.replace("RM", "").replace("MYR", "").strip()

    neg = False

    # parentheses negative
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # trailing sign
    if s.endswith("-"):
        neg = True
        s = s[:-1].strip()
    elif s.endswith("+"):
        s = s[:-1].strip()

    # leading sign
    if s.startswith("-"):
        neg = True
        s = s[1:].strip()

    s = s.replace(",", "").replace(" ", "")

    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return default

    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return default


def parse_amount(x: Any) -> Optional[float]:
    """Older parsers sometimes call parse_amount; keep it."""
    if x is None:
        return None
    s = normalize_text(x)
    if not s:
        return None
    if not _MONEY_TOKEN_RE.match(s.replace("RM", "").replace("MYR", "").strip()):
        # still attempt safe_float; it returns default but we want None on failure
        v = safe_float(s, default=float("nan"))
        return None if v != v else v
    v = safe_float(s, default=float("nan"))
    return None if v != v else v


# =============================================================================
# DATE NORMALIZATION
# =============================================================================
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DMY_SLASH_RE = re.compile(r"^(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{2,4})$")
_DMY_DASH_RE = re.compile(r"^(?P<d>\d{1,2})-(?P<m>\d{1,2})-(?P<y>\d{2,4})$")
# ✅ NEW: Alliance format ddmmyy / ddmmyyyy
_DDMMYY_RE = re.compile(r"^(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{2})$")
_DDMMYYYY_RE = re.compile(r"^(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{4})$")

# "01May2025" / "01May" etc. (some parsers might pass this in)
_DD_MON_RE = re.compile(r"^(?P<d>\d{1,2})(?P<mon>[A-Za-z]{3})(?P<y>\d{2,4})?$")

_MON_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _to_year(y: str) -> int:
    y = str(y)
    if len(y) == 2:
        return 2000 + int(y)
    return int(y)


def normalize_date(value: Any, *, default_year: Optional[int] = None) -> Optional[str]:
    """
    Normalize date into ISO 'YYYY-MM-DD', else None.

    Supports:
    - YYYY-MM-DD
    - DD/MM/YYYY, D/M/YY
    - DD-MM-YYYY
    - DDMMYY (Alliance)
    - DDMMYYYY (Alliance)
    - 01May2025 / 01May (year may use default_year)
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    s = normalize_text(value)
    if not s:
        return None

    if _ISO_RE.fullmatch(s):
        return s

    m = _DMY_SLASH_RE.match(s)
    if m:
        d = int(m.group("d"))
        mo = int(m.group("m"))
        y = _to_year(m.group("y"))
        try:
            return date(y, mo, d).isoformat()
        except Exception:
            return None

    m = _DMY_DASH_RE.match(s)
    if m:
        d = int(m.group("d"))
        mo = int(m.group("m"))
        y = _to_year(m.group("y"))
        try:
            return date(y, mo, d).isoformat()
        except Exception:
            return None

    m = _DDMMYYYY_RE.match(s)
    if m:
        d = int(m.group("d"))
        mo = int(m.group("m"))
        y = int(m.group("y"))
        try:
            return date(y, mo, d).isoformat()
        except Exception:
            return None

    m = _DDMMYY_RE.match(s)
    if m:
        d = int(m.group("d"))
        mo = int(m.group("m"))
        y = _to_year(m.group("y"))
        try:
            return date(y, mo, d).isoformat()
        except Exception:
            return None

    m = _DD_MON_RE.match(s)
    if m:
        d = int(m.group("d"))
        mon = m.group("mon").upper()
        mo = _MON_MAP.get(mon)
        y_raw = m.group("y")
        y = _to_year(y_raw) if y_raw else default_year
        if mo and y:
            try:
                return date(int(y), mo, d).isoformat()
            except Exception:
                return None

    # partial dd/mm with default_year
    if default_year is not None:
        m2 = re.match(r"^(?P<d>\d{1,2})/(?P<m>\d{1,2})$", s)
        if m2:
            d = int(m2.group("d"))
            mo = int(m2.group("m"))
            try:
                return date(int(default_year), mo, d).isoformat()
            except Exception:
                return None

    return None


def month_key(iso_date: str) -> Optional[str]:
    if not iso_date or not isinstance(iso_date, str):
        return None
    if not _ISO_RE.fullmatch(iso_date):
        return None
    return iso_date[:7]


# =============================================================================
# TRANSACTION STANDARDIZATION (THIS FIXES YOUR IMPORTERROR)
# =============================================================================
def _coerce_page(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def normalize_transactions(
    txns: List[Dict[str, Any]],
    *,
    default_bank: str = "",
    source_file: str = "",
) -> List[Dict[str, Any]]:
    """
    Normalize bank-specific extracted transactions into a standard schema:

      {
        "date": "YYYY-MM-DD" or None,
        "description": str,
        "debit": float,
        "credit": float,
        "balance": float or None,
        "page": int or None,
        "bank": str,
        "source_file": str,
        ... (preserve extra keys)
      }

    This keeps compatibility with the rest of your repo.
    """
    out: List[Dict[str, Any]] = []
    for t in (txns or []):
        if not isinstance(t, dict):
            continue

        row = dict(t)

        # Date normalization
        d = row.get("date")
        row["date"] = normalize_date(d) if d is not None else None

        # Description
        row["description"] = normalize_text(row.get("description", ""))

        # Support alt keys used by some parsers
        # - amount + dr_cr
        if row.get("debit") is None and row.get("credit") is None and row.get("amount") is not None:
            amt = safe_float(row.get("amount"), default=0.0)
            side = normalize_text(row.get("dr_cr", "")).upper()
            if side in ("DR", "DEBIT"):
                row["debit"] = abs(amt)
                row["credit"] = 0.0
            elif side in ("CR", "CREDIT"):
                row["debit"] = 0.0
                row["credit"] = abs(amt)

        row["debit"] = round(safe_float(row.get("debit", 0.0), default=0.0), 2)
        row["credit"] = round(safe_float(row.get("credit", 0.0), default=0.0), 2)

        bal = row.get("balance", None)
        row["balance"] = None if bal is None else round(safe_float(bal, default=0.0), 2)

        row["page"] = _coerce_page(row.get("page"))
        row["bank"] = normalize_text(row.get("bank") or default_bank)
        row["source_file"] = normalize_text(row.get("source_file") or source_file)

        out.append(row)

    return out


def dedupe_transactions(
    txns: List[Dict[str, Any]],
    *,
    key_fields: Tuple[str, ...] = ("date", "description", "debit", "credit", "balance", "page", "source_file"),
) -> List[Dict[str, Any]]:
    """Generic dedupe used by most banks."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for t in (txns or []):
        if not isinstance(t, dict):
            continue
        key = tuple(t.get(k) for k in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def dedupe_transactions_affin(txns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Affin statements often create near-duplicates (wrap lines / repeated headers).
    We dedupe using a slightly more tolerant key (ignore page if needed).
    """
    # First pass: strict
    strict = dedupe_transactions(
        txns,
        key_fields=("date", "description", "debit", "credit", "balance", "page", "source_file"),
    )
    # Second pass: tolerant (drop page)
    tolerant = dedupe_transactions(
        strict,
        key_fields=("date", "description", "debit", "credit", "balance", "source_file"),
    )
    return tolerant
