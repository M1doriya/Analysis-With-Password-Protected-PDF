"""
core_utils.py

Project-wide utilities used by Streamlit apps and bank parsers.

Goals:
1) Standardize input handling (PDF bytes)
2) Standardize transaction schema and types
3) Make date/amount parsing resilient across banks
4) Provide consistent monthly summary computation
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
    """
    Return PDF bytes from:
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
# TEXT / NUMBER NORMALIZATION
# =============================================================================
_WS_RE = re.compile(r"\s+")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]+")

def normalize_text(x: Any) -> str:
    """
    Convert to a clean one-line string:
    - strip
    - collapse whitespace
    - remove non-printable
    """
    if x is None:
        return ""
    s = str(x)
    s = _NON_PRINTABLE_RE.sub(" ", s)
    s = s.replace("\u00a0", " ")  # nbsp
    s = _WS_RE.sub(" ", s).strip()
    return s


def safe_float(x: Any, default: float = 0.0) -> float:
    """
    Parse common money-like strings into float.
    Accepts:
      "1,234.56"
      "-1,234.56"
      "(1,234.56)"  -> -1234.56
      "1,234.56-"   -> -1234.56
      "1,234.56+"   -> +1234.56
    Returns default on failure.
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

    # remove currency and spaces
    s = s.replace("RM", "").replace("MYR", "")
    s = s.replace(",", "").replace(" ", "")

    # allow leading sign
    if s.startswith("-"):
        neg = True
        s = s[1:]

    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return default

    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return default


# =============================================================================
# DATE NORMALIZATION
# =============================================================================
# Supported:
#  - ISO: 2025-05-31
#  - D/M/Y or DD/MM/YY: 31/5/2025, 31/05/25
#  - D-M-Y: 31-05-2025
#  - DMY compact: 150525 (DDMMYY), 15052025 (DDMMYYYY)
#  - "01May" style handled by specific bank parsers, but we still expose helpers
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DMY_SLASH_RE = re.compile(r"^(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{2,4})$")
_DMY_DASH_RE = re.compile(r"^(?P<d>\d{1,2})-(?P<m>\d{1,2})-(?P<y>\d{2,4})$")
_DDMMYY_RE = re.compile(r"^(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{2})$")
_DDMMYYYY_RE = re.compile(r"^(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{4})$")


def _to_year(y: str) -> int:
    y = str(y)
    if len(y) == 2:
        # assume 2000+
        return 2000 + int(y)
    return int(y)


def normalize_date(value: Any, *, default_year: Optional[int] = None) -> Optional[str]:
    """
    Normalize date into ISO 'YYYY-MM-DD', else None.

    default_year:
      used only when a parser passes a partial date (rare). This function mainly
      supports full dates. If you pass something like "31/05", handle it in bank parser.
    """
    if value is None:
        return None

    if isinstance(value, (datetime, date)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()

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

    # If something like "31/05" appears, bank parsers should resolve year using statement date.
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
    """
    Convert ISO date YYYY-MM-DD to month key YYYY-MM.
    """
    if not iso_date or not isinstance(iso_date, str):
        return None
    if not _ISO_RE.fullmatch(iso_date):
        return None
    return iso_date[:7]


# =============================================================================
# TRANSACTION STANDARDIZATION
# =============================================================================
def standardize_transaction(tx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Force a consistent schema, safe types, and rounding.
    Expected keys (best-effort):
      date, description, debit, credit, balance, page, bank, source_file
    """
    out: Dict[str, Any] = dict(tx or {})

    out["date"] = normalize_date(out.get("date")) or out.get("date")
    out["description"] = normalize_text(out.get("description", ""))
    out["debit"] = round(safe_float(out.get("debit", 0.0)), 2)
    out["credit"] = round(safe_float(out.get("credit", 0.0)), 2)
    out["balance"] = round(safe_float(out.get("balance", 0.0)), 2) if out.get("balance") is not None else None

    # keep page as int when possible
    try:
        out["page"] = int(out["page"]) if out.get("page") is not None else None
    except Exception:
        out["page"] = out.get("page")

    out["bank"] = normalize_text(out.get("bank", ""))
    out["source_file"] = normalize_text(out.get("source_file", ""))

    return out


def standardize_transactions(txs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [standardize_transaction(t) for t in (txs or [])]


def dedupe_transactions(
    txs: List[Dict[str, Any]],
    *,
    key_fields: Tuple[str, ...] = ("date", "description", "debit", "credit", "balance", "source_file"),
) -> List[Dict[str, Any]]:
    """
    Remove exact duplicates based on a stable tuple key.
    Keeps first occurrence.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for tx in txs or []:
        t = standardize_transaction(tx)
        key = tuple(t.get(k) for k in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


# =============================================================================
# MONTHLY SUMMARY (STANDARD)
# =============================================================================
def _compute_opening_from_first_tx(first_tx: Dict[str, Any]) -> Optional[float]:
    """
    Given first transaction and its balance + debit/credit, infer opening balance.
    opening + credit - debit = first_balance
    => opening = first_balance - credit + debit
    """
    bal = first_tx.get("balance")
    if bal is None:
        return None
    balf = safe_float(bal, default=0.0)
    cr = safe_float(first_tx.get("credit", 0.0), default=0.0)
    dr = safe_float(first_tx.get("debit", 0.0), default=0.0)
    return round(balf - cr + dr, 2)


def compute_monthly_summary(
    txs: List[Dict[str, Any]],
    *,
    include_opening_in_minmax: bool = True,
) -> List[Dict[str, Any]]:
    """
    Returns list of dict rows with:
      month, transaction_count, opening_balance, total_debit, total_credit, net_change,
      highest_balance, lowest_balance, source_files

    Notes:
    - opening_balance is inferred from the FIRST tx in that month using:
        opening = first_balance - credit + debit
      This is the safest generalized approach across banks when per-tx balance exists.
    - closing balance is last tx balance; net_change = closing - opening.
    - high/low uses tx balances (and optionally opening).
    """
    if not txs:
        return []

    txs_std = [t for t in standardize_transactions(txs) if isinstance(t.get("date"), str)]
    # only keep ISO dates
    txs_std = [t for t in txs_std if month_key(t["date"])]

    # sort by date then page as tie-breaker
    txs_std.sort(key=lambda t: (t.get("date") or "", t.get("page") or 0))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for tx in txs_std:
        mk = month_key(tx["date"])
        if not mk:
            continue
        grouped.setdefault(mk, []).append(tx)

    rows: List[Dict[str, Any]] = []
    for mk in sorted(grouped.keys()):
        items = grouped[mk]
        if not items:
            continue

        first = items[0]
        last = items[-1]

        opening = _compute_opening_from_first_tx(first)
        closing = last.get("balance")
        closing_f = safe_float(closing, default=0.0) if closing is not None else None

        total_debit = round(sum(safe_float(t.get("debit", 0.0)) for t in items), 2)
        total_credit = round(sum(safe_float(t.get("credit", 0.0)) for t in items), 2)

        # net change
        if opening is not None and closing_f is not None:
            net_change = round(closing_f - opening, 2)
        else:
            net_change = round(total_credit - total_debit, 2)

        balances = [safe_float(t.get("balance"), default=0.0) for t in items if t.get("balance") is not None]
        if include_opening_in_minmax and opening is not None:
            balances = [opening] + balances

        highest = round(max(balances), 2) if balances else None
        lowest = round(min(balances), 2) if balances else None

        src_files = sorted({normalize_text(t.get("source_file", "")) for t in items if t.get("source_file")})

        rows.append(
            {
                "month": mk,
                "transaction_count": len(items),
                "opening_balance": opening,
                "total_debit": total_debit,
                "total_credit": total_credit,
                "net_change": net_change,
                "highest_balance": highest,
                "lowest_balance": lowest,
                "source_files": src_files,
            }
        )

    return rows
