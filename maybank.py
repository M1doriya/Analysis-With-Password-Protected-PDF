import re
import fitz
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Regex patterns
# -----------------------------
DATE_DMY_SLASH_RE = re.compile(r"^(?P<d>\d{2})/(?P<m>\d{2})(?:/(?P<y>\d{2,4}))?$")
DATE_DMY_DASH_RE  = re.compile(r"^(?P<d>\d{2})-(?P<m>\d{2})(?:-(?P<y>\d{2,4}))?$")
STATEMENT_DATE_RE = re.compile(r"STATEMENT\s+DATE\s*:?\s*(\d{2})/(\d{2})/(\d{2,4})", re.I)

# Amount tokens usually look like: 1,630.00-  or  9,576.40+
MONEY_RE = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}[+-]?$")

FOOTER_KEYWORDS = (
    "ENDING BALANCE",
    "LEDGER BALANCE",
    "TOTAL DEBIT",
    "TOTAL CREDIT",
    "TOTAL DEBITS",
    "TOTAL CREDITS",
    "END OF STATEMENT",
    "PROFIT OUTSTANDING",
    "BAKI LEGAR",
    "BAKI AKHIR",
    "MUKA/",
    "PAGE",
    "NOMBOR AKAUN",
    "NOT PROTECTED BY PIDM",
    "PLEASE BE REMINDED",
    "NOTICE:",
    "NOTIS",
)

MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _open_doc(inp: Any) -> fitz.Document:
    """Open a PDF input robustly for Streamlit, bytes, file-like, or path."""
    if isinstance(inp, (bytes, bytearray)):
        return fitz.open(stream=bytes(inp), filetype="pdf")

    # Streamlit UploadedFile often supports getvalue()
    if hasattr(inp, "getvalue"):
        try:
            b = inp.getvalue()
            return fitz.open(stream=b, filetype="pdf")
        except Exception:
            pass

    # file-like object
    if hasattr(inp, "read"):
        try:
            pos = inp.tell()
        except Exception:
            pos = None

        b = inp.read()

        if pos is not None:
            try:
                inp.seek(pos)
            except Exception:
                pass

        return fitz.open(stream=b, filetype="pdf")

    # path string
    return fitz.open(inp)


def _parse_year_and_bank(doc: fitz.Document) -> Tuple[str, int]:
    bank = "Maybank"
    year = None

    for i in range(min(2, doc.page_count)):
        txt = (doc[i].get_text("text") or "").upper()

        if "MAYBANK ISLAMIC" in txt:
            bank = "Maybank Islamic"
        elif "MAYBANK" in txt:
            bank = "Maybank"

        m = STATEMENT_DATE_RE.search(txt)
        if m:
            y = m.group(3)
            year = (2000 + int(y)) if len(y) == 2 else int(y)
            break

    if year is None:
        year = datetime.now().year

    return bank, year


def _is_footer_or_header(line_text: str) -> bool:
    up = line_text.upper()
    return any(k in up for k in FOOTER_KEYWORDS)


def _money_token_value(tok: str) -> Tuple[float, Optional[str]]:
    """
    Returns (value, sign) where sign is '+' or '-' if present at end.
    """
    s = tok.strip()
    sign = None

    if s.endswith("+"):
        sign = "+"
        s = s[:-1]
    elif s.endswith("-"):
        sign = "-"
        s = s[:-1]

    s = s.replace(",", "")
    return float(s), sign


def _parse_date_token(token: str, default_year: int) -> Optional[str]:
    """
    Supports:
      DD/MM
      DD/MM/YY
      DD/MM/YYYY
      DD-MM
      DD-MM-YY
      DD-MM-YYYY
    """
    t = token.strip().upper()

    m = DATE_DMY_SLASH_RE.match(t)
    if m:
        d, mo, y = m.group("d"), m.group("m"), m.group("y")
        yy = default_year
        if y:
            yy = int(y)
            if yy < 100:
                yy = 2000 + yy
        return f"{yy:04d}-{int(mo):02d}-{int(d):02d}"

    m = DATE_DMY_DASH_RE.match(t)
    if m:
        d, mo, y = m.group("d"), m.group("m"), m.group("y")
        yy = default_year
        if y:
            yy = int(y)
            if yy < 100:
                yy = 2000 + yy
        return f"{yy:04d}-{int(mo):02d}-{int(d):02d}"

    return None


def _parse_split_date_tokens(items: List[dict]) -> Optional[str]:
    """
    Supports:
      DD MON YYYY   (e.g., 2 FEB 2025 / 02 FEB 2025)
    Only used if the PDF actually emits those tokens separately.
    """
    if len(items) < 3:
        return None

    d = items[0]["text"]
    mon = items[1]["text"]
    y = items[2]["text"]

    if not d.isdigit() or not y.isdigit():
        return None

    mon_u = mon.upper()
    if mon_u not in MONTH_MAP:
        return None

    return f"{int(y):04d}-{int(MONTH_MAP[mon_u]):02d}-{int(d):02d}"


def _cluster_lines(word_items: List[dict], y_tol: float = 3.0) -> List[Tuple[float, List[dict]]]:
    """
    Cluster words into 'visual lines' using y proximity.
    This is critical for Maybank PDFs because date/desc and amount/balance
    can be slightly misaligned in y.
    """
    if not word_items:
        return []

    word_items.sort(key=lambda r: (r["y"], r["x0"]))
    clusters: List[dict] = []

    for it in word_items:
        placed = False
        for c in clusters:
            if abs(it["y"] - c["y"]) <= y_tol:
                c["items"].append(it)
                # update centroid
                c["y"] = (c["y"] * (len(c["items"]) - 1) + it["y"]) / len(c["items"])
                placed = True
                break
        if not placed:
            clusters.append({"y": it["y"], "items": [it]})

    clusters.sort(key=lambda c: c["y"])

    out: List[Tuple[float, List[dict]]] = []
    for c in clusters:
        c["items"].sort(key=lambda r: r["x0"])
        out.append((c["y"], c["items"]))

    return out


def parse_transactions_maybank(pdf_input: Any, source_filename: str = "") -> List[Dict]:
    """
    Maybank (Conventional + Islamic) statement parser.

    Output schema matches other banks:
      date (YYYY-MM-DD), description, debit, credit, balance, page, bank, source_file
    """
    doc = _open_doc(pdf_input)
    bank_name, default_year = _parse_year_and_bank(doc)

    txs: List[Dict] = []
    prev_balance: Optional[float] = None

    # Context: Maybank sometimes omits the date for subsequent rows with the same date
    carry_date_iso: Optional[str] = None

    # Context: a date-only line may appear (rare) followed by the actual data row below
    pending_date_iso: Optional[str] = None
    pending_date_x_end: Optional[float] = None

    # For multi-line description continuation
    last_tx: Optional[Dict] = None
    last_desc_left: Optional[float] = None
    last_money_left: Optional[float] = None

    # Thresholds (tuned for Maybank PDFs)
    DATE_COL_RIGHT_FALLBACK = 85.0  # if date cell is blank, description starts after this

    def append_desc(line_items: List[dict]):
        nonlocal last_tx
        if not last_tx or last_desc_left is None or last_money_left is None:
            return

        parts = []
        for it in line_items:
            if it["is_money"]:
                continue
            if it["x0"] < last_desc_left:
                continue
            if it["x0"] >= last_money_left:
                continue
            parts.append(it["text"])

        if parts:
            last_tx["description"] = _norm_spaces(
                (last_tx.get("description", "") + " " + " ".join(parts)).strip()
            )

    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            words = page.get_text("words")

            word_items = []
            for w in words:
                txt = str(w[4]).strip()
                if not txt:
                    continue
                word_items.append({"y": float(w[1]), "x0": float(w[0]), "text": txt})

            for _, line_items in _cluster_lines(word_items, y_tol=3.0):
                line_text = _norm_spaces(" ".join(i["text"] for i in line_items))
                if not line_text:
                    continue

                if _is_footer_or_header(line_text):
                    pending_date_iso = None
                    pending_date_x_end = None
                    last_tx = None
                    last_desc_left = None
                    last_money_left = None
                    continue

                # Mark money tokens + find money column start
                money_positions = []
                for it in line_items:
                    it["is_money"] = bool(MONEY_RE.match(it["text"]))
                    if it["is_money"]:
                        money_positions.append(it["x0"])

                money_left = min(money_positions) if money_positions else None

                # Detect date token at start (DD/MM...) OR split date tokens (DD MON YYYY)
                date_iso: Optional[str] = None
                date_x_end: Optional[float] = None
                start_after_date = 0

                split = _parse_split_date_tokens(line_items[:3]) if len(line_items) >= 3 else None
                if split:
                    date_iso = split
                    date_x_end = line_items[2]["x0"] + 20.0
                    start_after_date = 3
                else:
                    if line_items:
                        maybe = _parse_date_token(line_items[0]["text"], default_year)
                        if maybe:
                            date_iso = maybe
                            date_x_end = line_items[0]["x0"] + 20.0
                            start_after_date = 1

                if date_iso:
                    carry_date_iso = date_iso  # always update date context

                # Date-only line (no money) -> pending date for the next line
                if date_iso and money_left is None:
                    # Keep only if it is "mostly just a date"
                    if len(line_items) <= start_after_date + 3:
                        pending_date_iso = date_iso
                        pending_date_x_end = date_x_end
                        last_tx = None
                        continue

                # Determine date for a transaction row
                effective_date = date_iso or pending_date_iso

                # Fix: blank-date rows (same date as previous transaction),
                # where date column is empty but amount/balance exist.
                if effective_date is None and money_left is not None and carry_date_iso:
                    first_x = line_items[0]["x0"] if line_items else 0.0
                    # If the first token is far right, date cell is likely blank
                    if first_x > 70.0:
                        effective_date = carry_date_iso
                        date_x_end = DATE_COL_RIGHT_FALLBACK

                # Transaction row
                if effective_date and money_left is not None:
                    money_tokens = [it["text"] for it in line_items if it["is_money"]]
                    if len(money_tokens) >= 2:
                        # balance is the last money token
                        bal_val, _ = _money_token_value(money_tokens[-1])

                        # Choose transaction amount: prefer last signed token before balance
                        amt_val: Optional[float] = None
                        amt_sign: Optional[str] = None

                        for t in reversed(money_tokens[:-1]):
                            v, sgn = _money_token_value(t)
                            if sgn in ("+", "-"):
                                amt_val, amt_sign = v, sgn
                                break

                        if amt_val is None:
                            v, sgn = _money_token_value(money_tokens[-2])
                            amt_val, amt_sign = v, sgn

                        # Description tokens by x-range
                        desc_left = date_x_end if date_x_end is not None else pending_date_x_end
                        if desc_left is None:
                            desc_left = DATE_COL_RIGHT_FALLBACK

                        desc_parts = []
                        for it in line_items:
                            if it["is_money"]:
                                continue
                            if it["x0"] < desc_left:
                                continue
                            if it["x0"] >= money_left:
                                continue
                            # skip the actual date tokens if split-date
                            if date_iso and it in line_items[:start_after_date]:
                                continue
                            desc_parts.append(it["text"])

                        description = _norm_spaces(" ".join(desc_parts))

                        debit = 0.0
                        credit = 0.0

                        # Primary rule: sign on the transaction amount
                        if amt_sign == "+":
                            credit = float(amt_val)
                        elif amt_sign == "-":
                            debit = float(amt_val)
                        else:
                            # Fallback: balance delta if sign missing
                            if prev_balance is not None:
                                delta = round(bal_val - prev_balance, 2)
                                if delta > 0:
                                    credit = abs(delta)
                                elif delta < 0:
                                    debit = abs(delta)
                            else:
                                debit = float(amt_val)

                        # Conservative OCR correction using balance delta
                        # (Fix digit swaps / missing decimals, without breaking correct values)
                        if prev_balance is not None:
                            delta = round(bal_val - prev_balance, 2)
                            expected = abs(delta)
                            parsed = credit if credit > 0 else debit
                            if expected > 0 and parsed > 0 and abs(expected - parsed) <= 500:
                                if delta > 0:
                                    credit = expected
                                    debit = 0.0
                                elif delta < 0:
                                    debit = expected
                                    credit = 0.0

                        tx = {
                            "date": effective_date,
                            "description": description,
                            "debit": round(float(debit), 2),
                            "credit": round(float(credit), 2),
                            "balance": round(float(bal_val), 2),
                            "page": page_index + 1,
                            "bank": bank_name,
                            "source_file": source_filename,
                        }
                        txs.append(tx)

                        # Continuation boundaries
                        last_tx = tx
                        last_desc_left = desc_left
                        last_money_left = money_left

                        prev_balance = bal_val

                        # Clear pending date after use
                        if date_iso is None and pending_date_iso is not None:
                            pending_date_iso = None
                            pending_date_x_end = None

                        continue

                # Description continuation lines
                if last_tx is not None and money_left is None and date_iso is None:
                    append_desc(line_items)

    finally:
        doc.close()

    # Dedupe (ignore description differences; keep unique by numeric signature)
    seen = set()
    out: List[Dict] = []
    for t in txs:
        key = (t["date"], t["debit"], t["credit"], t["balance"], t["page"], t["source_file"])
        if key in seen:
            continue
        seen.add(key)
        out.append(t)

    return out
