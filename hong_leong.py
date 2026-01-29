import re
from datetime import datetime


# =========================================================
# Regex
# =========================================================

# Dates: 26-09-2025
DATE_TOKEN_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")

# Money tokens may appear as:
#   1,234.56
#   1,234.56-
#   1,234.56+
#   (34,923.86)
#   (34,923.86)-
MONEY_TOKEN_RE = re.compile(r"^\(?(?P<num>[\d,]+\.\d{2})\)?(?P<sign>[+-])?$")

# Detect statement header/opening lines (NOT a real transaction)
# These rows cause ending balance to shift into the next month if included.
HEADER_OPENING_RE = re.compile(
    r"(Statement Period|Balance from previous statement|Date\s*/\s*Tarikh)",
    re.I
)

# Optional: detect explicit OD mention in PDF text.
OD_KEYWORDS_RE = re.compile(
    r"\b(overdraft|od\s+facility|od\s+limit|overdrawn|excess\s+limit|interest\s+on\s+overdraft|excess\s+interest)\b",
    re.I
)


# =========================================================
# MAIN ENTRY (USED BY app.py)
# =========================================================

def parse_hong_leong(pdf, filename):
    """
    Fixes implemented INSIDE the parser:

    1) Skips statement header/opening rows so monthly ending_balance doesn't shift:
       - Rows containing "Statement Period" / "Balance from previous statement" / "Date / Tarikh"
         with debit=0 and credit=0 are not emitted as transactions.

    2) Supports parentheses negative amounts: (34,923.86)

    3) Anchors running_balance to printed balance whenever extracted, preventing drift/false OD.
    """
    transactions = []

    opening_balance = extract_opening_balance(pdf)
    running_balance = float(opening_balance)

    # Track last statement-anchored balance (initialize to opening to prevent early drift negatives)
    last_stmt_balance = float(opening_balance)

    # Keyword-based OD detection (backup signal)
    overdraft_possible = pdf_mentions_overdraft(pdf)

    for page_num, page in enumerate(pdf.pages, start=1):
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if not words:
            continue

        rows = group_words_by_row(words, tolerance=3)

        # Detect Deposit/Withdrawal/Balance column x positions (per page)
        col_x = detect_amount_columns(rows)

        i = 0
        while i < len(rows):
            row = rows[i]

            date = extract_date(row)
            if not date:
                i += 1
                continue

            if is_total_row(row):
                i += 1
                continue

            desc_tokens = []
            amount_tokens = []

            # Current row
            desc_tokens.extend(extract_desc_tokens(row))
            amount_tokens.extend(extract_amount_tokens(row, col_x))

            # Continuation rows until next date
            j = i + 1
            while j < len(rows) and extract_date(rows[j]) is None:
                if is_total_row(rows[j]):
                    break
                desc_tokens.extend(extract_desc_tokens(rows[j]))
                amount_tokens.extend(extract_amount_tokens(rows[j], col_x))
                j += 1

            # Existing behavior: classify credit/debit from amount tokens
            credit, debit = classify_amounts_by_columns(amount_tokens, col_x)

            # Extract statement balance from balance column token if present
            stmt_balance = extract_statement_balance(amount_tokens, col_x)

            # Build a description string for header detection
            desc_text = clean_description(desc_tokens)

            # ---------------------------------------------------------
            # CRITICAL FIX #1: Skip header/opening rows (non-transactions)
            # ---------------------------------------------------------
            # These rows usually have a date, 0 debit, 0 credit, and contain
            # statement period / balance from previous statement / date-tarikh.
            # Including them causes ending_balance to shift into the next month.
            if credit == 0.0 and debit == 0.0 and HEADER_OPENING_RE.search(desc_text or ""):
                # Still allow anchoring if they carry a printed balance
                if stmt_balance is not None:
                    running_balance = float(stmt_balance)
                    last_stmt_balance = float(stmt_balance)
                    if stmt_balance < 0:
                        overdraft_possible = True
                i = j
                continue

            # If nothing meaningful found, skip
            if credit == 0.0 and debit == 0.0 and stmt_balance is None:
                i = j
                continue

            # Compute fallback running balance
            computed_balance = round(running_balance + credit - debit, 2)

            # If statement balance exists and is negative, OD is possible regardless of keyword scan
            if stmt_balance is not None and float(stmt_balance) < 0:
                overdraft_possible = True

            # Anchor to statement balance if available
            if stmt_balance is not None:
                running_balance = float(stmt_balance)
                last_stmt_balance = float(stmt_balance)
            else:
                # No statement balance captured on this row:
                # If computed would go negative but statement has no OD evidence, do NOT surface fake negative.
                if (not overdraft_possible) and computed_balance < 0:
                    running_balance = float(last_stmt_balance)
                else:
                    running_balance = float(computed_balance)

            transactions.append({
                "date": date,
                "description": desc_text,
                "debit": round(float(debit), 2),
                "credit": round(float(credit), 2),
                "balance": round(float(running_balance), 2),
                "page": int(page_num),
                "bank": "Hong Leong Islamic Bank",
                "source_file": filename
            })

            i = j

    return transactions


# =========================================================
# OPENING BALANCE
# =========================================================

def extract_opening_balance(pdf):
    text = pdf.pages[0].extract_text() or ""
    m = re.search(
        r"Balance from previous statement\s+([\d,]+\.\d{2})",
        text,
        re.IGNORECASE
    )
    if not m:
        raise ValueError("Opening balance not found")
    return float(m.group(1).replace(",", ""))


def pdf_mentions_overdraft(pdf) -> bool:
    for p in pdf.pages:
        t = (p.extract_text() or "")
        if OD_KEYWORDS_RE.search(t):
            return True
    return False


# =========================================================
# ROW GROUPING (Y AXIS)
# =========================================================

def group_words_by_row(words, tolerance=3):
    rows = []
    for w in words:
        placed = False
        for row in rows:
            if abs(row[0]["top"] - w["top"]) <= tolerance:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])

    for row in rows:
        row.sort(key=lambda x: x["x0"])
    return rows


# =========================================================
# COLUMN DETECTION (Deposit / Withdrawal / Balance)
# =========================================================

def detect_amount_columns(rows):
    deposit_x = withdrawal_x = balance_x = None

    for row in rows:
        joined = " ".join(w["text"].strip().lower() for w in row)

        # header typically contains all three labels
        if "deposit" in joined and "withdrawal" in joined and "balance" in joined:
            for w in row:
                t = w["text"].strip().lower()
                if t == "deposit":
                    deposit_x = w["x0"]
                elif t == "withdrawal":
                    withdrawal_x = w["x0"]
                elif t == "balance":
                    balance_x = w["x0"]
            break

    # sensible fallbacks if header text isn't captured on some pages
    if deposit_x is None:
        deposit_x = 320.0
    if withdrawal_x is None:
        withdrawal_x = 410.0
    if balance_x is None:
        balance_x = 520.0

    return {
        "deposit_x": float(deposit_x),
        "withdrawal_x": float(withdrawal_x),
        "balance_x": float(balance_x),
    }


# =========================================================
# DATE DETECTION
# =========================================================

def extract_date(row):
    for w in row:
        if DATE_TOKEN_RE.fullmatch(w["text"]):
            return datetime.strptime(w["text"], "%d-%m-%Y").strftime("%Y-%m-%d")
    return None


# =========================================================
# TOKEN EXTRACTION
# =========================================================

def parse_money_token(s: str) -> float:
    """
    Parse:
      1,234.56
      1,234.56-
      1,234.56+
      (34,923.86) or (34,923.86)-  -> negative
    Returns signed float.
    """
    s = s.strip()
    m = MONEY_TOKEN_RE.match(s)
    if not m:
        raise ValueError("Not a money token")

    num = float(m.group("num").replace(",", ""))
    sign = m.group("sign")

    is_paren_neg = s.startswith("(") and ")" in s
    is_minus = (sign == "-")

    if is_paren_neg or is_minus:
        return -num
    return num


def extract_amount_tokens(row, col_x):
    """
    Only treat money-looking tokens as amounts if they are positioned
    in the right-side amount columns area (near Deposit/Withdrawal/Balance).
    """
    out = []
    min_amount_x = col_x["deposit_x"] - 25

    for w in row:
        t = w["text"].strip()
        if MONEY_TOKEN_RE.match(t):
            if float(w["x0"]) >= float(min_amount_x):
                val = parse_money_token(t)
                out.append({"x": float(w["x0"]), "value": float(val)})

    return out


def extract_desc_tokens(row):
    out = []
    for w in row:
        t = w["text"].strip()
        if not t:
            continue
        if DATE_TOKEN_RE.fullmatch(t):
            continue
        if is_noise(t):
            continue
        out.append(t)
    return out


# =========================================================
# AMOUNT CLASSIFICATION USING COLUMN X (ignore balance column)
# =========================================================

def classify_amounts_by_columns(amount_words, col_x):
    credit = 0.0
    debit = 0.0

    dep = float(col_x["deposit_x"])
    wdr = float(col_x["withdrawal_x"])
    bal = float(col_x["balance_x"])

    for a in amount_words:
        x = float(a["x"])
        val = abs(float(a["value"]))

        dist_dep = abs(x - dep)
        dist_wdr = abs(x - wdr)
        dist_bal = abs(x - bal)

        if dist_dep <= dist_wdr and dist_dep <= dist_bal:
            credit += val
        elif dist_wdr <= dist_dep and dist_wdr <= dist_bal:
            debit += val
        else:
            # balance column -> ignore for debit/credit
            pass

    return round(credit, 2), round(debit, 2)


# =========================================================
# STATEMENT BALANCE EXTRACTION
# =========================================================

def extract_statement_balance(amount_words, col_x, tol=80.0):
    """
    Extract the statement printed Balance token from amount_words.

    Most reliable heuristic:
      - balance is the RIGHTMOST money token in the balance column area.

    Returns signed float or None.
    """
    if not amount_words:
        return None

    bal_x = float(col_x["balance_x"])

    # Strong rule: anything clearly in/after balance column -> pick rightmost
    balance_area = [a for a in amount_words if float(a["x"]) >= (bal_x - 10)]
    if balance_area:
        rightmost = max(balance_area, key=lambda a: float(a["x"]))
        return round(float(rightmost["value"]), 2)

    # Fallback: proximity scoring if balance_x estimate is off
    dep_x = float(col_x["deposit_x"])
    wdr_x = float(col_x["withdrawal_x"])

    candidates = []
    for a in amount_words:
        x = float(a["x"])
        v = float(a["value"])
        if abs(x - bal_x) <= float(tol):
            if abs(x - bal_x) <= abs(x - dep_x) and abs(x - bal_x) <= abs(x - wdr_x):
                candidates.append((abs(x - bal_x), -x, v))

    if not candidates:
        return None

    candidates.sort()
    return round(float(candidates[0][2]), 2)


# =========================================================
# FILTERS / CLEANUP
# =========================================================

def is_total_row(row):
    text = " ".join(w["text"] for w in row)
    return bool(re.search(
        r"Total Withdrawals|Total Deposits|Closing Balance|Important Notices",
        text,
        re.IGNORECASE
    ))


def is_noise(text):
    return bool(re.search(
        r"Protected by PIDM|Dilindungi oleh PIDM|Hong Leong Islamic Bank|hlisb\.com\.my|Menara Hong Leong|CURRENT ACCOUNT",
        text,
        re.IGNORECASE
    ))


def clean_description(parts):
    s = " ".join(parts)
    s = re.sub(r"\s+", " ", s).strip()
    return s
