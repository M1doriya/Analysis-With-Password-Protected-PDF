# agro_bank.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

# =========================================================
# Regex / constants
# =========================================================

# Dates in Agrobank statements: 31/05/25
DATE_RE = re.compile(r"^\d{1,2}/\d{2}/\d{2}$")

# Amount tokens:
# - 1,234.56
# - 1,234.56-
# - .92
# - .50-
AMOUNT_RE = re.compile(r"^(?P<num>(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d{2})(?P<sign>-)?$")

# Common zero formats
ZERO_RE = re.compile(r"^(?:0|0?\.00)(?:-)?$")


def _to_float(amount_token: str) -> float:
    """Parse Agrobank amount tokens, supporting leading-dot (.92) and trailing '-' for negatives."""
    s = (amount_token or "").strip()
    if not s:
        return 0.0
    neg = s.endswith("-")
    if neg:
        s = s[:-1]
    if s.startswith("."):
        s = "0" + s
    s = s.replace(",", "")
    v = float(s)
    return -v if neg else v


def extract_agrobank_summary_totals(pdf: pdfplumber.PDF) -> Tuple[Optional[float], Optional[float]]:
    """Extract TOTAL DEBIT / TOTAL CREDIT from the statement footer (most reliable source)."""
    total_debit = None
    total_credit = None

    for page in reversed(pdf.pages):
        text = page.extract_text() or ""
        for line in text.splitlines():
            u = line.upper()

            if "TOTAL DEBIT" in u:
                m = re.search(r"([\d,]*\d?\.\d{2})", line)
                if m:
                    total_debit = _to_float(m.group(1))

            if "TOTAL CREDIT" in u:
                m = re.search(r"([\d,]*\d?\.\d{2})", line)
                if m:
                    total_credit = _to_float(m.group(1))

        if total_debit is not None and total_credit is not None:
            break

    return total_debit, total_credit


def parse_agro_bank(pdf: pdfplumber.PDF, source_file: str) -> List[Dict[str, Any]]:
    """
    Agrobank parser (pdfplumber)

    Root cause of wrong monthly balances:
    - The old code skipped BEGINNING BALANCE and CLOSING BALANCE lines.
      Many monthly summaries take the first/last extracted transaction balance as opening/ending.
      That makes month-end balances drift (e.g., June ending balance).

    Fix:
    - Emit BEGINNING BALANCE and CLOSING BALANCE as synthetic rows (is_balance_marker=True).
    - Keep balance-delta inference for debit/credit (works well for Agrobank).
    - If a month-end adjustment is only reflected in the closing line, delta will capture it there.
    """

    transactions: List[Dict[str, Any]] = []
    previous_balance: Optional[float] = None

    summary_debit, summary_credit = extract_agrobank_summary_totals(pdf)

    for page_num, page in enumerate(pdf.pages, start=1):
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        words = sorted(words, key=lambda w: (w["top"], w["x0"]))

        i = 0
        while i < len(words):
            token = (words[i].get("text") or "").strip()

            if DATE_RE.fullmatch(token):
                y_ref = words[i]["top"]
                same_line = [w for w in words if abs(w["top"] - y_ref) <= 2]

                amounts = [(w["x0"], (w["text"] or "").strip()) for w in same_line if AMOUNT_RE.fullmatch((w["text"] or "").strip())]
                amounts.sort(key=lambda x: x[0])

                if not amounts:
                    i += 1
                    continue

                # last money token is the running BALANCE for Agrobank
                balance = _to_float(amounts[-1][1])

                description = " ".join(
                    w["text"] for w in same_line
                    if not DATE_RE.fullmatch((w["text"] or "").strip())
                    and not AMOUNT_RE.fullmatch((w["text"] or "").strip())
                    and not ZERO_RE.fullmatch((w["text"] or "").strip())
                ).strip()

                iso_date = datetime.strptime(token, "%d/%m/%y").strftime("%Y-%m-%d")
                desc_upper = description.upper()

                # ---------------------------------------------
                # BEGINNING / CLOSING BALANCE as marker rows
                # ---------------------------------------------
                if "BEGINNING BALANCE" in desc_upper:
                    transactions.append({
                        "date": iso_date,
                        "description": "BEGINNING BALANCE",
                        "debit": None,
                        "credit": None,
                        "balance": round(balance, 2),
                        "page": page_num,
                        "bank": "Agrobank",
                        "source_file": source_file,
                        "is_balance_marker": True,
                    })
                    previous_balance = balance
                    i += 1
                    continue

                if "CLOSING BALANCE" in desc_upper:
                    debit = credit = None
                    if previous_balance is not None:
                        delta = balance - previous_balance
                        if delta > 0.0001:
                            credit = round(delta, 2)
                        elif delta < -0.0001:
                            debit = round(abs(delta), 2)

                    transactions.append({
                        "date": iso_date,
                        "description": "CLOSING BALANCE",
                        "debit": debit,
                        "credit": credit,
                        "balance": round(balance, 2),
                        "page": page_num,
                        "bank": "Agrobank",
                        "source_file": source_file,
                        "is_balance_marker": True,
                    })
                    previous_balance = balance
                    i += 1
                    continue

                # ---------------------------------------------
                # NORMAL TRANSACTION (infer debit/credit from delta)
                # ---------------------------------------------
                debit = credit = None
                if previous_balance is not None:
                    delta = balance - previous_balance
                    if delta > 0.0001:
                        credit = round(delta, 2)
                    elif delta < -0.0001:
                        debit = round(abs(delta), 2)

                transactions.append({
                    "date": iso_date,
                    "description": description,
                    "debit": debit,
                    "credit": credit,
                    "balance": round(balance, 2),
                    "page": page_num,
                    "bank": "Agrobank",
                    "source_file": source_file,
                })

                previous_balance = balance

            i += 1

    # ---------------------------------------------
    # SUMMARY VALIDATION (optional marker)
    # ---------------------------------------------
    computed_debit = round(sum(t.get("debit") or 0 for t in transactions), 2)
    computed_credit = round(sum(t.get("credit") or 0 for t in transactions), 2)

    mismatch = False
    if summary_debit is not None and abs(computed_debit - summary_debit) > 0.01:
        mismatch = True
    if summary_credit is not None and abs(computed_credit - summary_credit) > 0.01:
        mismatch = True

    for t in transactions:
        t["summary_check"] = "#" if mismatch else ""

    return transactions
