# bank_muamalat.py

import re
from datetime import datetime

DATE_RE = re.compile(r"\d{1,2}/\d{2}/\d{2}")
AMOUNT_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d{2}")
ZERO_RE = re.compile(r"^0?\.00$")


def parse_transactions_bank_muamalat(pdf, source_file):
    """
    Bank Muamalat parser.

    Key features:
    - Uses date anchor + same-line logic
    - Right-most amount = balance
    - Debit/Credit decided by balance delta
    - Outputs ISO date (YYYY-MM-DD) to FIX monthly summary
    """

    transactions = []
    previous_balance = None

    for page_num, page in enumerate(pdf.pages, start=1):

        words = page.extract_words(
            use_text_flow=True,
            keep_blank_chars=False
        )

        # sort visually (top → bottom, left → right)
        words = sorted(words, key=lambda w: (w["top"], w["x0"]))

        i = 0
        while i < len(words):

            text = words[i]["text"]

            # -------------------------
            # DATE ANCHOR
            # -------------------------
            if DATE_RE.fullmatch(text):

                y_ref = words[i]["top"]

                same_line = [
                    w for w in words
                    if abs(w["top"] - y_ref) <= 2
                ]

                # Clean description
                description = " ".join(
                    w["text"] for w in same_line
                    if not DATE_RE.fullmatch(w["text"])
                    and not AMOUNT_RE.fullmatch(w["text"])
                    and not ZERO_RE.fullmatch(w["text"])
                ).strip()

                # Extract numeric amounts
                amounts = [
                    (w["x0"], w["text"])
                    for w in same_line
                    if AMOUNT_RE.fullmatch(w["text"])
                    and not ZERO_RE.fullmatch(w["text"])
                ]

                if not amounts:
                    i += 1
                    continue

                amounts = sorted(amounts, key=lambda x: x[0])

                # Right-most amount = balance
                current_balance = float(amounts[-1][1].replace(",", ""))

                # Transaction amount = last non-balance value
                txn_amount = None
                if len(amounts) > 1:
                    txn_amount = float(amounts[-2][1].replace(",", ""))

                debit = credit = None

                # -------------------------
                # DEBIT / CREDIT BY BALANCE DELTA
                # -------------------------
                if txn_amount is not None and previous_balance is not None:
                    delta = current_balance - previous_balance

                    if delta > 0.0001:
                        credit = abs(delta)
                    elif delta < -0.0001:
                        debit = abs(delta)
                else:
                    # fallback for first row
                    desc_upper = description.upper()
                    if desc_upper.startswith("CR") or "PROFIT PAID" in desc_upper:
                        credit = txn_amount
                    else:
                        debit = txn_amount

                # -------------------------
                # ISO DATE OUTPUT (CRITICAL FIX)
                # -------------------------
                iso_date = datetime.strptime(text, "%d/%m/%y").strftime("%Y-%m-%d")

                transactions.append({
                    "date": iso_date,  # ✅ FIXED FORMAT
                    "description": description,
                    "debit": debit,
                    "credit": credit,
                    "balance": current_balance,
                    "page": page_num,
                    "bank": "Bank Muamalat",
                    "source_file": source_file
                })

                previous_balance = current_balance

            i += 1

    return transactions
