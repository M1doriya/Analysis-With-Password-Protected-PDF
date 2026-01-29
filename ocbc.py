# ocbc.py
# OCBC Bank (Malaysia) - Current Account statement parser
#
# FIX:
#   Some statements have NO transaction lines (only Balance B/F + Transaction Summary = 0/0).
#   In that case, emit a single balance-only row dated to the statement end date so the month
#   appears in Streamlit monthly summary.

from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from core_utils import normalize_text, safe_float


# --- Patterns ---
TX_START_RE = re.compile(
    r"^(?P<day>\d{2})\s+"
    r"(?P<mon>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+"
    r"(?P<year>\d{4})\s+"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)

BAL_BF_RE = re.compile(r"\bBalance\s+B/F\b\s+(?P<bal>-?[\d,]+\.\d{2})", re.IGNORECASE)

# Statement period line example (from your PDF):
# "Statement Date / Tarikh Penyata : 01 APR 2023 TO 30 APR 2023"
STATEMENT_PERIOD_RE = re.compile(
    r"Statement\s+Date\s*/\s*Tarikh\s+Penyata\s*:\s*"
    r"(?P<d1>\d{2})\s+(?P<m1>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(?P<y1>\d{4})\s+TO\s+"
    r"(?P<d2>\d{2})\s+(?P<m2>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(?P<y2>\d{4})",
    re.IGNORECASE,
)

MONEY_RE = re.compile(r"^-?(\d{1,3}(?:,\d{3})*|\d+)\.\d{2}$")

STOP_LINES = (
    "TRANSACTION",
    "SUMMARY",
    "NO. OF WITHDRAWALS",
    "NO. OF DEPOSITS",
    "TOTAL WITHDRAWALS",
    "TOTAL DEPOSITS",
    "HOLD AMOUNT",
    "LATE LOCAL CHEQUE",
    "PAGE ",
    "STATEMENT OF CURRENT ACCOUNT",
    "PENYATA AKAUN SEMASA",
    "TRANSACTION DATE",
    "TARIKH TRANSAKSI",
    "TRANSACTION DESCRIPTION",
    "HURAIAN TRANSAKSI",
)

# classification hints
CREDIT_HINTS = (" CR ", "CR /IB", "CR INWARD", "CREDIT")
DEBIT_HINTS = (" DR ", "DR /IB", "DEBIT", "DUITNOW SC", "DEBIT AS ADVISED")


def _to_iso_date(day: str, mon: str, year: str) -> str:
    mon = mon.upper()
    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    mm = month_map.get(mon, "01")
    return f"{year}-{mm}-{day}"


def _extract_statement_end_date_iso(text: str) -> Optional[str]:
    """Extract the statement end date ISO from the statement period header."""
    if not text:
        return None
    m = STATEMENT_PERIOD_RE.search(text)
    if not m:
        return None
    return _to_iso_date(m.group("d2"), m.group("m2"), m.group("y2"))


def _extract_amount_and_balance_from_line(rest: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    From 'rest' (after date), extract:
      - tx_amount (usually the penultimate money token)
      - balance (last money token)
      - desc_text (rest with trailing numeric columns removed)
    """
    tokens = rest.split()
    money_idx = [i for i, t in enumerate(tokens) if MONEY_RE.match(t)]
    if len(money_idx) < 2:
        return None, None, rest

    balance = safe_float(tokens[money_idx[-1]])
    tx_amount = safe_float(tokens[money_idx[-2]])

    cut = money_idx[-2]
    desc_text = " ".join(tokens[:cut]).strip()
    return tx_amount, balance, desc_text


def _is_noise_line(line: str) -> bool:
    up = line.upper().strip()
    if not up:
        return True
    return any(k in up for k in STOP_LINES)


def parse_transactions_ocbc(pdf_input: Any, source_file: str = "") -> List[Dict]:
    """
    Standard interface used by app.py:
      input: pdf bytes (preferred) OR file-like
      output: list of tx dicts with canonical keys
    """
    if isinstance(pdf_input, (bytes, bytearray)):
        pdf = pdfplumber.open(BytesIO(bytes(pdf_input)))
    else:
        pdf = pdfplumber.open(pdf_input)

    bank_name = "OCBC Bank"
    transactions: List[Dict] = []

    prev_balance: Optional[float] = None
    statement_end_iso: Optional[str] = None
    current_tx: Optional[Dict] = None

    try:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue

            # capture statement end date once (needed for "no transactions" months)
            if statement_end_iso is None:
                statement_end_iso = _extract_statement_end_date_iso(text)

            # find Balance B/F once
            if prev_balance is None:
                bf = BAL_BF_RE.search(text)
                if bf:
                    prev_balance = safe_float(bf.group("bal"))

            for raw_line in text.splitlines():
                line = normalize_text(raw_line)
                if not line:
                    continue

                # Stop processing transaction area when summary starts
                if "TRANSACTION" in line.upper() and "SUMMARY" in line.upper():
                    current_tx = None
                    break

                m = TX_START_RE.match(line)
                if m:
                    day, mon, year = m.group("day"), m.group("mon"), m.group("year")
                    rest = m.group("rest")

                    date_iso = _to_iso_date(day, mon, year)
                    tx_amount, balance, desc_head = _extract_amount_and_balance_from_line(rest)

                    if tx_amount is None or balance is None:
                        current_tx = None
                        continue

                    desc_upper = desc_head.upper()
                    debit = 0.0
                    credit = 0.0

                    # 1) keyword classification
                    if any(h in f" {desc_upper} " for h in CREDIT_HINTS) and not any(h in f" {desc_upper} " for h in (" DR ", "DR /IB")):
                        credit = abs(tx_amount)
                    elif any(h in f" {desc_upper} " for h in DEBIT_HINTS):
                        debit = abs(tx_amount)
                    # 2) balance-delta fallback
                    elif prev_balance is not None:
                        delta = round(balance - prev_balance, 2)
                        if abs(delta - tx_amount) <= 0.05:
                            credit = abs(tx_amount)
                        elif abs(delta + tx_amount) <= 0.05:
                            debit = abs(tx_amount)
                        else:
                            if delta > 0:
                                credit = abs(delta)
                            elif delta < 0:
                                debit = abs(delta)

                    tx = {
                        "date": date_iso,
                        "description": desc_head,
                        "debit": round(float(debit), 2),
                        "credit": round(float(credit), 2),
                        "balance": round(float(balance), 2),
                        "page": page_idx,
                        "bank": bank_name,
                        "source_file": source_file,
                    }
                    transactions.append(tx)
                    current_tx = tx
                    prev_balance = balance
                    continue

                # Continuation lines (multi-line description)
                if current_tx is not None and not _is_noise_line(line):
                    # avoid numeric-only lines
                    if not MONEY_RE.match(line.replace(",", "")):
                        current_tx["description"] = normalize_text(current_tx["description"] + " " + line)

        # ---- FIX: no transactions case (like your April PDF) ----
        if not transactions and prev_balance is not None:
            # Use statement end date so monthly summary buckets correctly
            date_for_row = statement_end_iso or "2000-01-01"
            transactions.append(
                {
                    "date": date_for_row,
                    "description": "NO TRANSACTIONS (BALANCE B/F)",
                    "debit": 0.0,
                    "credit": 0.0,
                    "balance": round(float(prev_balance), 2),
                    "page": None,
                    "bank": bank_name,
                    "source_file": source_file,
                    "is_statement_balance": True,
                }
            )

        return transactions

    finally:
        try:
            pdf.close()
        except Exception:
            pass
