
# bank_rakyat.py
# Bank Rakyat – Balance-driven, summary-aware parser (FINAL)

import re
from datetime import datetime


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_amount(val):
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return None


def parse_date(raw):
    try:
        return datetime.strptime(raw, "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        return None


# ---------------------------------------------------------
# Extract summary section (source of truth)
# ---------------------------------------------------------

def extract_summary(full_text):
    """
    Extracts opening, total debit, total credit, closing.
    Works even if values appear BELOW labels.
    """

    nums = [clean_amount(x) for x in re.findall(r"[-]?\d[\d,]*\.\d{2}", full_text)]
    nums = [n for n in nums if n is not None]

    summary = {
        "opening": None,
        "total_debit": None,
        "total_credit": None,
        "closing": None,
    }

    # Explicit patterns (preferred)
    m = re.search(r"(Opening Balance|Baki Permulaan)[^\d\-]*([-]?\d[\d,]*\.\d{2})", full_text, re.I | re.S)
    if m:
        summary["opening"] = clean_amount(m.group(2))

    m = re.search(r"(Closing Balance|Baki Penutup)[^\d\-]*([-]?\d[\d,]*\.\d{2})", full_text, re.I | re.S)
    if m:
        summary["closing"] = clean_amount(m.group(2))

    # Fallback: Bank Rakyat summary row ALWAYS has 4 numbers
    # [opening, total debit, total credit, closing]
    if len(nums) >= 4:
        summary["opening"] = summary["opening"] or nums[-4]
        summary["total_debit"] = nums[-3]
        summary["total_credit"] = nums[-2]
        summary["closing"] = summary["closing"] or nums[-1]

    return summary


# ---------------------------------------------------------
# Extract raw transaction rows (order-independent)
# ---------------------------------------------------------

def extract_transactions(pdf):
    rows = []

    for page_no, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        for line in text.splitlines():

            date_match = re.search(r"\d{2}/\d{2}/\d{4}", line)
            if not date_match:
                continue

            amounts = re.findall(r"[-]?\d[\d,]*\.\d{2}", line)
            if not amounts:
                continue

            balance = clean_amount(amounts[-1])
            if balance is None:
                continue

            iso_date = parse_date(date_match.group())
            if not iso_date:
                continue

            desc = line.replace(date_match.group(), "")
            desc = desc.replace(amounts[-1], "")
            desc = " ".join(desc.split())

            rows.append({
                "date": iso_date,
                "description": desc,
                "balance": balance,
                "page": page_no,
            })

    return rows


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------

def parse_bank_rakyat(pdf, source_filename=""):
    # Read entire document text
    full_text = ""
    for p in pdf.pages:
        full_text += (p.extract_text() or "") + "\n"

    summary = extract_summary(full_text)
    raw_rows = extract_transactions(pdf)

    if not raw_rows:
        return []

    # Sort chronologically
    raw_rows.sort(key=lambda x: (x["date"], x["page"]))

    # Determine opening balance (BEST METHOD)
    opening = summary["opening"]

    if opening is None and summary["closing"] is not None:
        opening = (
            summary["closing"]
            - (summary["total_credit"] or 0)
            + (summary["total_debit"] or 0)
        )

    results = []
    prev_balance = opening

    for row in raw_rows:
        debit = credit = 0.0

        if prev_balance is not None:
            delta = round(row["balance"] - prev_balance, 2)
            if delta > 0:
                credit = delta
            elif delta < 0:
                debit = abs(delta)

        prev_balance = row["balance"]

        results.append({
            "date": row["date"],
            "description": row["description"],
            "debit": debit,
            "credit": credit,
            "balance": row["balance"],
            "page": row["page"],
            "bank": "Bank Rakyat",
            "source_file": source_filename,
        })

    return results
