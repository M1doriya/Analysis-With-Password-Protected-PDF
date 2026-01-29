# ambank.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pdfplumber

# =========================================================
# Regex patterns
# =========================================================

# Transaction starts look like: 01May <rest...>
TX_START_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

# Money tokens
MONEY_ANYWHERE_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}(?!\d)")

# Statement date range (month derivation)
STMT_RANGE_RE = re.compile(
    r"STATEMENT\s+DATE.*?:\s*(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)

# Account summary labels (English/Malay)
OPENING_LBL_RE = re.compile(r"(OPENING\s+BALANCE|BAKI\s+PEMBUKAAN)", re.IGNORECASE)
CLOSING_LBL_RE = re.compile(r"(CLOSING\s+BALANCE|BAKI\s+PENUTUPAN)", re.IGNORECASE)
TOTAL_DEBIT_LBL_RE = re.compile(r"(TOTAL\s+DEBITS?|JUMLAH\s+DEBIT)", re.IGNORECASE)
TOTAL_CREDIT_LBL_RE = re.compile(r"(TOTAL\s+CREDITS?|JUMLAH\s+KREDIT)", re.IGNORECASE)


_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# =========================================================
# Helpers
# =========================================================

def _safe_float_money(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if not MONEY_ANYWHERE_RE.fullmatch(s):
        return None
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def _normalize_lines_keep_order(text: str) -> List[str]:
    lines: List[str] = []
    for raw in (text or "").splitlines():
        ln = re.sub(r"\s+", " ", raw).strip()
        if ln:
            lines.append(ln)
    return lines


def _find_amount_near_label(lines: List[str], label_re: re.Pattern) -> Optional[float]:
    """
    Robust for AmBank Account Summary where amounts sometimes appear on the line ABOVE the label.

    Strategy:
    - Find line index where label appears.
    - Try to find money token on the same line.
    - If none, scan upward for first line containing a money token (within a small window).
    - If still none, scan downward within a small window.
    """
    idxs = [i for i, ln in enumerate(lines) if label_re.search(ln)]
    if not idxs:
        return None

    i = idxs[0]

    # same line
    m = MONEY_ANYWHERE_RE.findall(lines[i])
    if m:
        return _safe_float_money(m[-1])

    # look up a few lines (most common layout)
    for j in range(i - 1, max(-1, i - 6), -1):
        m2 = MONEY_ANYWHERE_RE.findall(lines[j])
        if m2:
            return _safe_float_money(m2[-1])

    # look down a few lines (fallback)
    for j in range(i + 1, min(len(lines), i + 6)):
        m3 = MONEY_ANYWHERE_RE.findall(lines[j])
        if m3:
            return _safe_float_money(m3[-1])

    return None


def extract_ambank_statement_totals(pdf: pdfplumber.PDF, source_file: str = "") -> Dict[str, Optional[float]]:
    """
    Source-of-truth extraction from AmBank statement Account Summary (page 1).

    Returns:
      {
        statement_month: "YYYY-MM" | None,
        opening_balance: float | None,
        ending_balance: float | None,
        total_debit: float | None,
        total_credit: float | None,
        source_file: str
      }
    """
    out: Dict[str, Optional[float]] = {
        "statement_month": None,
        "opening_balance": None,
        "ending_balance": None,
        "total_debit": None,
        "total_credit": None,
        "source_file": source_file,
    }
    if not pdf.pages:
        return out

    text1 = pdf.pages[0].extract_text(x_tolerance=1) or ""
    lines = _normalize_lines_keep_order(text1)

    # statement month from date range end date
    m = STMT_RANGE_RE.search(text1)
    if m:
        try:
            end_dt = datetime.strptime(m.group(2), "%d/%m/%Y")
            out["statement_month"] = end_dt.strftime("%Y-%m")
        except Exception:
            pass

    # Extract amounts by label proximity (works whether amount is on same line or above)
    out["opening_balance"] = _find_amount_near_label(lines, OPENING_LBL_RE)
    out["ending_balance"] = _find_amount_near_label(lines, CLOSING_LBL_RE)
    out["total_debit"] = _find_amount_near_label(lines, TOTAL_DEBIT_LBL_RE)
    out["total_credit"] = _find_amount_near_label(lines, TOTAL_CREDIT_LBL_RE)

    return out


def _to_iso_date(day: str, mon: str, year: int) -> Optional[str]:
    mm = _MONTH_MAP.get((mon or "").upper())
    if not mm:
        return None
    try:
        dd = int(day)
        dt = datetime(year, mm, dd)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _extract_last_balance_from_text(s: str) -> Optional[float]:
    if not s:
        return None
    # take the last money token as the running balance
    m = re.findall(r"\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2}", s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except Exception:
        return None


def _finalize_tx(
    *,
    date_iso: str,
    buf: List[str],
    page_num: int,
    filename: str,
    prev_balance: Optional[float],
    seq: int,
) -> Tuple[Optional[Dict], Optional[float]]:
    joined = " ".join([b for b in buf if b]).strip()
    if not joined:
        return None, prev_balance

    balance = _extract_last_balance_from_text(joined)
    if balance is None:
        return None, prev_balance

    # remove trailing balance token from description
    desc = re.sub(r"\s*(\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})\s*(?:DR|CR)?\s*$", "", joined, flags=re.I).strip()
    desc = re.sub(r"\s+", " ", desc).strip()

    # delta inference (still used for line items display; not used for monthly totals)
    debit = 0.0
    credit = 0.0
    if prev_balance is not None:
        delta = round(balance - prev_balance, 2)
        if delta > 0:
            credit = delta
        elif delta < 0:
            debit = abs(delta)

    tx = {
        "date": date_iso,
        "description": desc,
        "debit": round(float(debit), 2),
        "credit": round(float(credit), 2),
        "balance": round(float(balance), 2),
        "page": int(page_num),
        "seq": int(seq),
        "bank": "AmBank Islamic",
        "source_file": filename,
    }
    return tx, balance


def _parse_transactions_from_lines(
    lines: List[str],
    *,
    page_num: int,
    filename: str,
    detected_year: int,
    prev_balance: Optional[float],
    seq_start: int,
) -> Tuple[List[Dict], Optional[float], int]:
    txs: List[Dict] = []
    buf: List[str] = []
    cur_date_iso: Optional[str] = None
    seq = seq_start

    def flush():
        nonlocal prev_balance, seq, buf, cur_date_iso
        if cur_date_iso is None:
            buf = []
            return
        tx, new_prev = _finalize_tx(
            date_iso=cur_date_iso,
            buf=buf,
            page_num=page_num,
            filename=filename,
            prev_balance=prev_balance,
            seq=seq,
        )
        if tx:
            txs.append(tx)
            prev_balance = new_prev
            seq += 1
        buf = []
        cur_date_iso = None

    for ln in lines:
        m = TX_START_RE.match(ln)
        if m:
            flush()
            cur_date_iso = _to_iso_date(m.group("day"), m.group("mon"), detected_year)
            rest = (m.group("rest") or "").strip()
            buf = [rest] if (cur_date_iso and rest) else []
        else:
            if cur_date_iso is not None:
                buf.append(ln)

    flush()
    return txs, prev_balance, seq


def parse_ambank(pdf: pdfplumber.PDF, filename: str) -> List[Dict]:
    """
    Line items parser (kept lightweight).
    Monthly totals MUST come from extract_ambank_statement_totals(), not from these inferred deltas.
    """
    # year from statement range end date (best)
    detected_year = datetime.utcnow().year
    t0 = pdf.pages[0].extract_text(x_tolerance=1) or ""
    m = STMT_RANGE_RE.search(t0)
    if m:
        try:
            end_dt = datetime.strptime(m.group(2), "%d/%m/%Y")
            detected_year = end_dt.year
        except Exception:
            pass

    # Opening anchor (optional; not required for statement totals)
    statement_totals = extract_ambank_statement_totals(pdf, filename)
    prev_balance = statement_totals.get("opening_balance")

    transactions: List[Dict] = []
    seq = 0

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text(x_tolerance=1) or ""
        if not text.strip():
            continue
        lines = _normalize_lines_keep_order(text)
        # remove obvious header noise
        cleaned = [ln for ln in lines if "ACCOUNT SUMMARY" not in ln.upper()]

        page_txs, prev_balance, seq = _parse_transactions_from_lines(
            cleaned,
            page_num=page_num,
            filename=filename,
            detected_year=int(detected_year),
            prev_balance=prev_balance,
            seq_start=seq,
        )
        transactions.extend(page_txs)

    # deterministic ordering
    transactions = sorted(transactions, key=lambda t: (t.get("date") or "", int(t.get("page") or 0), int(t.get("seq") or 0)))
    return transactions
