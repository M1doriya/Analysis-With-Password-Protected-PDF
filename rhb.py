import re
import fitz  # PyMuPDF
import pdfplumber
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple


# ======================================================
# Helper: read PDF bytes safely (Streamlit / file / path)
# ======================================================
def _read_pdf_bytes(pdf_input: Any) -> bytes:
    """Return PDF bytes from bytes, Streamlit UploadedFile, file-like, or filesystem path."""
    if isinstance(pdf_input, (bytes, bytearray)):
        return bytes(pdf_input)

    # Streamlit UploadedFile
    if hasattr(pdf_input, "getvalue"):
        data = pdf_input.getvalue()
        if data:
            return data

    # file-like object
    if hasattr(pdf_input, "read"):
        try:
            pdf_input.seek(0)
        except Exception:
            pass
        data = pdf_input.read()
        if data:
            return data

    # path string
    if isinstance(pdf_input, str):
        with open(pdf_input, "rb") as f:
            return f.read()

    raise ValueError("Unable to read PDF bytes")


# -----------------------------
# Shared parsing helpers
# -----------------------------
_MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

# Money tokens in RHB statements often look like:
#   27,286.00
#   746,858.49-
#   0.00
_MONEY_TOKEN_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})*\.\d{2}[+-]?$|^[+-]?\d+\.\d{2}[+-]?$")


def _money_to_float(token: str) -> Optional[float]:
    if token is None:
        return None
    s = str(token).strip().replace(" ", "")
    if not s:
        return None

    # parenthesis negative
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()

    trailing = None
    if s.endswith("+"):
        trailing = "+"
        s = s[:-1]
    elif s.endswith("-"):
        trailing = "-"
        s = s[:-1]

    s = s.replace(",", "")
    try:
        v = float(s)
    except Exception:
        return None

    if trailing == "-":
        v = -abs(v)
    elif trailing == "+":
        v = abs(v)
    return float(v)


def _extract_year_from_statement_period(text: str) -> Optional[int]:
    """Extract year from common RHB 'Statement Period / Tempoh Penyata' header lines."""
    if not text:
        return None

    # e.g. "Statement Period / Tempoh Penyata : 1 Jan 25 – 31 Jan 25"
    m = re.search(
        r"Statement\s+Period.*?:\s*\d{1,2}\s+[A-Za-z]{3}\s+(?P<y1>\d{2,4})\s*[-–—]\s*"
        r"\d{1,2}\s+[A-Za-z]{3}\s+(?P<y2>\d{2,4})",
        text,
        re.IGNORECASE,
    )
    if m:
        y = m.group("y2") or m.group("y1")
        return int(y) if len(y) == 4 else 2000 + int(y)

    # weaker fallback: first "DD Mon YY/ YYYY" near Statement Period
    m = re.search(
        r"Statement\s+Period.*?:.*?\b\d{1,2}\s+[A-Za-z]{3}\s+(?P<y>\d{2,4})\b",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        y = m.group("y")
        return int(y) if len(y) == 4 else 2000 + int(y)

    return None


def _guess_bank_name(header_text_upper: str) -> str:
    if "RHB ISLAMIC" in header_text_upper:
        return "RHB Islamic Bank"
    return "RHB Bank"


# ======================================================
# 1) RHB "ACCOUNT STATEMENT / PENYATA AKAUN" (Jan/Feb/etc with space)
#    - This is the format in your uploaded file
#    - Negative balances are shown with trailing '-' (e.g., 746,858.49-)
#    - Multi-line descriptions continue on the following lines without a date
# ======================================================
def _parse_rhb_account_statement_text(pdf_bytes: bytes, source_filename: str) -> List[Dict]:
    transactions: List[Dict] = []

    DATE_START_RE = re.compile(r"^(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3})\b\s+(?P<rest>.*)$")
    NOISE_LINE_RE = re.compile(
        r"^(?:"
        r"ACCOUNT\s+ACTIVITY|DEPOSIT\s+ACCOUNT|DEPOSIT\s+ACCOUNT\s+SUMMARY|STATEMENT\s+PERIOD|"
        r"IMPORTANT\s+NOTES|IMPORTANT\s+ANNOUNCEMENTS|PAGE\s+NO\.?|RHB\s+BANK|"
        r"MEMBER\s+OF\s+PIDM|PROTECTED\s+BY\s+PIDM|DILINDUNGI\s+OLEH\s+PIDM|"
        r"PRODUCT\s+NAME|ACCOUNT\s+NO\.?|CURRENCY|DATE\s+DESCRIPTION|"
        r"CHEQUE\s+\/\s+SERIAL|DEBIT|CREDIT|BALANCE"
        r")\b",
        re.IGNORECASE,
    )

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        header = pdf.pages[0].extract_text(x_tolerance=1) or ""
        header_up = header.upper()

        # Heuristic: only run this parser if the statement looks like the account-statement format
        if "ACCOUNT STATEMENT" not in header_up and "PENYATA" not in header_up:
            return []

        year = _extract_year_from_statement_period(header) or datetime.now().year
        bank_name = _guess_bank_name(header_up)

        prev_balance: Optional[float] = None
        last_tx: Optional[Dict] = None

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1) or ""
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]

            for line in lines:
                if NOISE_LINE_RE.match(line):
                    last_tx = None
                    continue

                # Totals / summary counters
                if re.match(r"^Total\s+Count\b", line, re.IGNORECASE):
                    last_tx = None
                    continue

                m = DATE_START_RE.match(line)
                if m:
                    dd = int(m.group("day"))
                    mon = m.group("mon").upper()
                    if mon not in _MONTH_MAP:
                        last_tx = None
                        continue

                    date_iso = f"{year:04d}-{_MONTH_MAP[mon]}-{dd:02d}"

                    tokens = line.split()
                    rest_tokens = tokens[2:]  # drop day + mon

                    money_idx = [i for i, t in enumerate(rest_tokens) if _MONEY_TOKEN_RE.match(t)]
                    if not money_idx:
                        last_tx = None
                        continue

                    bal_token = rest_tokens[money_idx[-1]]
                    balance = _money_to_float(bal_token)
                    if balance is None:
                        last_tx = None
                        continue

                    # Description is everything before the numeric columns start
                    desc_tokens = rest_tokens[:money_idx[0]]
                    description = " ".join(desc_tokens).strip()

                    # Opening/closing balance lines (do not emit as transactions)
                    up_desc = description.upper()
                    if "B/F" in up_desc:
                        prev_balance = balance
                        last_tx = None
                        continue
                    if "C/F" in up_desc:
                        prev_balance = balance
                        last_tx = None
                        continue

                    # If we still don't have an anchor balance, we cannot infer debit/credit reliably
                    if prev_balance is None:
                        prev_balance = balance
                        last_tx = None
                        continue

                    delta = round(balance - prev_balance, 2)
                    debit = round(abs(delta), 2) if delta < 0 else 0.0
                    credit = round(delta, 2) if delta > 0 else 0.0

                    tx = {
                        "date": date_iso,
                        "description": description,
                        "debit": debit,
                        "credit": credit,
                        "balance": round(float(balance), 2),
                        "page": page_num,
                        "bank": bank_name,
                        "source_file": source_filename,
                    }
                    transactions.append(tx)

                    prev_balance = balance
                    last_tx = tx
                    continue

                # Continuation line (multi-line description)
                if last_tx is not None:
                    # Avoid appending lines that are just numbers
                    compact = line.replace(" ", "")
                    if _MONEY_TOKEN_RE.match(compact):
                        continue
                    if len(line) >= 300:
                        # likely a disclaimer paragraph; don't pollute descriptions
                        continue
                    last_tx["description"] = re.sub(r"\s+", " ", (last_tx["description"] + " " + line)).strip()

    return transactions


# ======================================================
# 2) RHB ISLAMIC — legacy text-based format (kept, but fixed)
# ======================================================
def _parse_rhb_islamic_text(pdf_bytes: bytes, source_filename: str) -> List[Dict]:
    transactions: List[Dict] = []
    previous_balance: Optional[float] = None

    balance_re = re.compile(r"(?P<bal>[\d,]+\.\d{2}[+-]?)\s*$")
    date_re = re.compile(r"(?P<d>\d{1,2})\s+(?P<m>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        header = pdf.pages[0].extract_text(x_tolerance=1) or ""
        year = _extract_year_from_statement_period(header) or datetime.now().year

        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue

            for line in text.splitlines():
                bal_match = balance_re.search(line.strip())
                date_match = date_re.search(line)
                if not bal_match or not date_match:
                    continue

                balance = _money_to_float(bal_match.group("bal"))
                if balance is None:
                    continue

                if re.search(r"\bB/F\b|\bC/F\b", line):
                    previous_balance = balance
                    continue

                if previous_balance is None:
                    previous_balance = balance
                    continue

                day = int(date_match.group("d"))
                month = date_match.group("m")
                date_iso = datetime.strptime(f"{day:02d} {month} {year}", "%d %b %Y").strftime("%Y-%m-%d")

                delta = round(balance - previous_balance, 2)
                debit = round(abs(delta), 2) if delta < 0 else 0.0
                credit = round(delta, 2) if delta > 0 else 0.0

                desc = balance_re.sub("", line)
                desc = desc.replace(date_match.group(0), "")
                desc = re.sub(r"\s+", " ", desc).strip()

                transactions.append(
                    {
                        "date": date_iso,
                        "description": desc,
                        "debit": debit,
                        "credit": credit,
                        "balance": round(balance, 2),
                        "page": page_index,
                        "bank": "RHB Islamic Bank",
                        "source_file": source_filename,
                    }
                )

                previous_balance = balance

    return transactions


# ======================================================
# 3) RHB CONVENTIONAL — older text-based format (kept, but fixed)
# ======================================================
def _parse_rhb_conventional_text(pdf_bytes: bytes, source_filename: str) -> List[Dict]:
    transactions: List[Dict] = []
    previous_balance: Optional[float] = None

    balance_re = re.compile(r"(?P<bal>[\d,]+\.\d{2}[+-]?)\s*$")
    # supports "05Jan" and "05 Jan"
    date_re = re.compile(r"(?P<d>\d{1,2})\s*(?P<m>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        header = pdf.pages[0].extract_text(x_tolerance=1) or ""
        year = _extract_year_from_statement_period(header) or datetime.now().year

        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue

            for line in text.splitlines():
                bal_m = balance_re.search(line.strip())
                date_m = date_re.search(line)
                if not bal_m or not date_m:
                    continue

                balance = _money_to_float(bal_m.group("bal"))
                if balance is None:
                    continue

                if previous_balance is None:
                    previous_balance = balance
                    continue

                day = int(date_m.group("d"))
                month = date_m.group("m")
                date_iso = datetime.strptime(f"{day:02d} {month} {year}", "%d %b %Y").strftime("%Y-%m-%d")

                delta = round(balance - previous_balance, 2)
                debit = round(abs(delta), 2) if delta < 0 else 0.0
                credit = round(delta, 2) if delta > 0 else 0.0

                desc = balance_re.sub("", line)
                desc = desc.replace(date_m.group(0), "")
                desc = re.sub(r"\s+", " ", desc).strip()

                transactions.append(
                    {
                        "date": date_iso,
                        "description": desc,
                        "debit": debit,
                        "credit": credit,
                        "balance": round(balance, 2),
                        "page": page_index,
                        "bank": "RHB Bank",
                        "source_file": source_filename,
                    }
                )

                previous_balance = balance

    return transactions


# ======================================================
# 4) RHB REFLEX — layout based (kept as-is)
# ======================================================
def _parse_rhb_reflex_layout(pdf_bytes: bytes, source_filename: str) -> List[Dict]:
    transactions: List[Dict] = []

    DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    MONEY_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})*|\d)?\.\d{2}[+-]?")

    def norm_date(text: str) -> str:
        return datetime.strptime(text, "%d-%m-%Y").strftime("%Y-%m-%d")

    def extract_opening_balance() -> Optional[float]:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if "Beginning Balance" in text:
                    m = re.search(r"([\d,]+\.\d{2})([+-])?", text)
                    if m:
                        amount = float(m.group(1).replace(",", ""))
                        if m.group(2) == "-":
                            amount = -amount
                        return amount
        return None

    previous_balance = extract_opening_balance()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        for page_index, page in enumerate(doc, start=1):
            words = page.get_text("words")
            rows = [
                {"x": w[0], "y": round(w[1], 1), "text": w[4].strip()}
                for w in words
                if w[4].strip()
            ]
            rows.sort(key=lambda r: (r["y"], r["x"]))
            used_y = set()

            for r in rows:
                if not DATE_RE.match(r["text"]):
                    continue

                y = r["y"]
                if y in used_y:
                    continue

                line = [w for w in rows if abs(w["y"] - y) <= 1.5]
                line.sort(key=lambda w: w["x"])

                money = [w for w in line if MONEY_RE.match(w["text"])]
                if len(money) < 2:
                    continue

                bal_text = money[-1]["text"].replace(",", "")
                is_negative = bal_text.endswith("-")
                bal_val = float(bal_text.replace("-", "").replace("+", ""))

                if is_negative:
                    bal_val = -bal_val

                debit = credit = 0.0
                if previous_balance is not None:
                    delta = round(bal_val - previous_balance, 2)
                    if delta < 0:
                        debit = abs(delta)
                    elif delta > 0:
                        credit = delta

                description_parts = [
                    w["text"]
                    for w in line
                    if w not in money and not DATE_RE.match(w["text"]) and not w["text"].isdigit()
                ]

                transactions.append(
                    {
                        "date": norm_date(r["text"]),
                        "description": " ".join(description_parts)[:200],
                        "debit": round(debit, 2),
                        "credit": round(credit, 2),
                        "balance": round(bal_val, 2),
                        "page": page_index,
                        "bank": "RHB Bank",
                        "source_file": source_filename,
                    }
                )

                previous_balance = bal_val
                used_y.add(y)

    finally:
        doc.close()

    return transactions


def parse_transactions_rhb(pdf_input: Any, source_filename: str) -> List[Dict]:
    """Main entry used by app.py: returns list of canonical tx dicts."""
    pdf_bytes = _read_pdf_bytes(pdf_input)

    # Order matters: try the Account Statement format FIRST (covers your uploaded PDF)
    for parser in (
        _parse_rhb_account_statement_text,
        _parse_rhb_islamic_text,
        _parse_rhb_conventional_text,
        _parse_rhb_reflex_layout,
    ):
        try:
            tx = parser(pdf_bytes, source_filename)
            if tx:
                return tx
        except Exception:
            continue

    return []
