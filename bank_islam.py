# bank_islam.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None

from PIL import ImageEnhance, ImageOps


# =========================================================
# BANK ISLAM – FORMAT 1 (TABLE-BASED)
# =========================================================
def parse_bank_islam_format1(pdf, source_file):
    transactions: List[Dict[str, Any]] = []

    def extract_amount(text):
        if not text:
            return None
        s = re.sub(r"\s+", "", str(text))
        m = re.search(r"(-?[\d,]+\.\d{2})", s)
        return float(m.group(1).replace(",", "")) if m else None

    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table()
        if not table:
            continue

        for row in table:
            row = list(row) if row else []
            while len(row) < 12:
                row.append(None)

            (
                no,
                txn_date,
                customer_eft,
                txn_code,
                description,
                ref_no,
                branch,
                debit_raw,
                credit_raw,
                balance_raw,
                sender_recipient,
                payment_details,
            ) = row[:12]

            if not txn_date or not re.search(r"\d{2}/\d{2}/\d{4}", str(txn_date)):
                continue

            try:
                date = datetime.strptime(
                    re.search(r"\d{2}/\d{2}/\d{4}", str(txn_date)).group(),
                    "%d/%m/%Y",
                ).date().isoformat()
            except Exception:
                continue

            debit = extract_amount(debit_raw) or 0.0
            credit = extract_amount(credit_raw) or 0.0
            balance = extract_amount(balance_raw) or 0.0

            # Recovery from description (kept as in your original)
            if debit == 0.0 and credit == 0.0:
                recovered = extract_amount(description)
                if recovered:
                    desc = str(description).upper()
                    if "CR" in desc or "CREDIT" in desc or "IN" in desc:
                        credit = recovered
                    else:
                        debit = recovered

            description_clean = " ".join(
                str(x).replace("\n", " ").strip()
                for x in [no, txn_code, description, sender_recipient, payment_details]
                if x and str(x).lower() != "nan"
            )

            transactions.append(
                {
                    "date": date,
                    "description": description_clean,
                    "debit": round(debit, 2),
                    "credit": round(credit, 2),
                    "balance": round(balance, 2),
                    "page": page_num,
                    "bank": "Bank Islam",
                    "source_file": source_file,
                    "format": "format1",
                }
            )

    return transactions


# =========================================================
# BANK ISLAM – FORMAT 2 (TEXT / STATEMENT-BASED)
# =========================================================
MONEY_RE = re.compile(r"\(?-?[\d,]+\.\d{2}\)?")
DATE_AT_START_RE = re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{2,4})\b")
BAL_BF_RE = re.compile(r"BAL\s+B/F", re.IGNORECASE)


def _to_float(val):
    if not val:
        return None
    neg = val.startswith("(") and val.endswith(")")
    val = val.strip("()").replace(",", "")
    try:
        num = float(val)
        return -num if neg else num
    except ValueError:
        return None


def _parse_date(d: str) -> Optional[str]:
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(d.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def parse_bank_islam_format2(pdf, source_file):
    transactions: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

        for line in lines:
            upper = line.upper()

            if BAL_BF_RE.search(upper):
                money = MONEY_RE.findall(line)
                if money:
                    prev_balance = _to_float(money[-1])
                continue

            m_date = DATE_AT_START_RE.match(line)
            if not m_date or prev_balance is None:
                continue

            date = _parse_date(m_date.group(1))
            if not date:
                continue

            money_raw = MONEY_RE.findall(line)
            money_vals = [_to_float(x) for x in money_raw if _to_float(x) is not None]
            if not money_vals:
                continue

            balance = money_vals[-1]

            delta = round(balance - prev_balance, 2)
            credit = delta if delta > 0 else 0.0
            debit = abs(delta) if delta < 0 else 0.0
            prev_balance = balance

            desc = line[len(m_date.group(1)) :].strip()
            for tok in money_raw:
                desc = desc.replace(tok, "").strip()

            transactions.append(
                {
                    "date": date,
                    "description": desc,
                    "debit": round(debit, 2),
                    "credit": round(credit, 2),
                    "balance": round(balance, 2),
                    "page": page_num,
                    "bank": "Bank Islam",
                    "source_file": source_file,
                    "format": "format2_balance_delta",
                }
            )

    return transactions


# =========================================================
# FORMAT 3 – eSTATEMENT
# =========================================================
def parse_bank_islam_format3(pdf, source_file):
    transactions: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    DATE_RE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})")
    MONEY_RE3 = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d{2})")
    BAL_BF_RE3 = re.compile(r"BAL\s+B/F", re.IGNORECASE)

    def to_float(x):
        return float(x.replace(",", ""))

    def parse_date(d):
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return datetime.strptime(d, fmt).date().isoformat()
            except ValueError:
                continue
        return None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text(x_tolerance=1) or ""
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

        for line in lines:
            if BAL_BF_RE3.search(line):
                nums = MONEY_RE3.findall(line)
                if nums:
                    prev_balance = to_float(nums[-1])
                continue

            date_match = DATE_RE.match(line)
            if date_match and prev_balance is not None:
                raw_date = date_match.group(1)
                nums = MONEY_RE3.findall(line)
                if len(nums) >= 2:
                    balance = to_float(nums[-1])
                    desc = line.replace(raw_date, "").strip()
                    for n in nums:
                        desc = desc.replace(n, "").strip()

                    delta = round(balance - prev_balance, 2)

                    transactions.append(
                        {
                            "date": parse_date(raw_date),
                            "description": desc,
                            "debit": abs(delta) if delta < 0 else 0.0,
                            "credit": delta if delta > 0 else 0.0,
                            "balance": balance,
                            "page": page_num,
                            "bank": "Bank Islam",
                            "source_file": source_file,
                            "format": "format3_estatement",
                        }
                    )
                    prev_balance = balance

    return transactions


# =========================================================
# FORMAT 4 – eSTATEMENT
# =========================================================
def parse_bank_islam_format4(pdf, source_file):
    transactions: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    DATE_RE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})")
    MONEY_RE4 = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d{2})")
    BAL_BF_RE4 = re.compile(r"BAL\s+B/IF", re.IGNORECASE)

    def to_float(x):
        return float(x.replace(",", ""))

    def parse_date(d):
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return datetime.strptime(d, fmt).date().isoformat()
            except ValueError:
                continue
        return None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text(x_tolerance=1) or ""
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

        for line in lines:
            if BAL_BF_RE4.search(line):
                nums = MONEY_RE4.findall(line)
                if nums:
                    prev_balance = to_float(nums[-1])
                continue

            date_match = DATE_RE.match(line)
            if date_match and prev_balance is not None:
                raw_date = date_match.group(1)
                nums = MONEY_RE4.findall(line)
                if nums:
                    current_balance = to_float(nums[-1])
                    delta = round(current_balance - prev_balance, 2)
                    desc = line.replace(raw_date, "").strip()
                    for n in nums:
                        desc = desc.replace(n, "").strip()

                    transactions.append(
                        {
                            "date": parse_date(raw_date),
                            "description": desc,
                            "debit": abs(delta) if delta < 0 else 0.0,
                            "credit": delta if delta > 0 else 0.0,
                            "balance": current_balance,
                            "page": page_num,
                            "bank": "Bank Islam",
                            "source_file": source_file,
                            "format": "format4_normalized",
                        }
                    )
                    prev_balance = current_balance

    return transactions


# =========================================================
# OCR PATH – BALANCE DELTA (with February fix)
# =========================================================
_OCR_DATE_RE = re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{2,4})\b")
_OCR_MONEY_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}")
_OCR_STATEMENT_DATE_RE = re.compile(
    r"(?:STATEMENT\s*DATE|TARIKH\s*PENYATA)\s*:?\s*(\d{1,2})/(\d{1,2})/(\d{2,4})",
    re.IGNORECASE,
)


def _ocr_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def _ocr_image(page, resolution: int = 400):
    img = page.to_image(resolution=resolution).original
    img = ImageOps.grayscale(img)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    return img


def _ocr_text_page_multi(page) -> str:
    if pytesseract is None:
        return ""
    img = _ocr_image(page, resolution=400)
    t4 = pytesseract.image_to_string(img, config="--psm 4") or ""
    t6 = pytesseract.image_to_string(img, config="--psm 6") or ""
    return t4 + "\n" + t6


def _extract_statement_month_year_via_ocr(pdf) -> Optional[Tuple[int, int]]:
    if pytesseract is None or not getattr(pdf, "pages", None):
        return None
    text = _ocr_text_page_multi(pdf.pages[0]) or ""
    m = _OCR_STATEMENT_DATE_RE.search(text)
    if not m:
        return None
    mm = int(m.group(2))
    yy_raw = m.group(3)
    yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)
    if 1 <= mm <= 12 and 2000 <= yy <= 2100:
        return (yy, mm)
    return None


def _extract_summary_totals_via_ocr(pdf) -> Tuple[Optional[float], Optional[float]]:
    if pytesseract is None or not getattr(pdf, "pages", None):
        return (None, None)

    text = _ocr_text_page_multi(pdf.pages[0])
    text_norm = re.sub(r"\s+", " ", (text or "")).upper()

    def find_total(label: str) -> Optional[float]:
        m = re.search(
            rf"{label}\s+(?:\d+\s+)?((?:\d{{1,3}}(?:,\d{{3}})*)\.\d{{2}})",
            text_norm,
        )
        return _ocr_float(m.group(1)) if m else None

    return find_total("TOTAL DEBIT"), find_total("TOTAL CREDIT")


def _extract_opening_balance_via_ocr(pdf) -> Optional[float]:
    if pytesseract is None or not getattr(pdf, "pages", None):
        return None

    text = _ocr_text_page_multi(pdf.pages[0])
    text_norm = re.sub(r"\s+", " ", (text or "")).upper()

    m = re.search(
        r"BAL\s*B/F\s+((?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2})",
        text_norm,
    )
    return _ocr_float(m.group(1)) if m else None


def _parse_date_dmy(raw: str) -> Optional[datetime]:
    raw = raw.strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _collect_date_balance_candidates_from_ocr(
    pdf, stmt_year_month: Optional[Tuple[int, int]]
) -> Dict[str, List[Tuple[float, str, int]]]:
    """
    Returns {date_iso: [(balance, line_text, page_num), ...]} possibly with multiple balances per date.
    """
    candidates: Dict[str, List[Tuple[float, str, int]]] = {}

    for page_num, page in enumerate(pdf.pages, start=1):
        text = _ocr_text_page_multi(page)
        lines = [re.sub(r"\s+", " ", l).strip() for l in (text or "").splitlines() if l.strip()]

        for line in lines:
            dm = _OCR_DATE_RE.match(line)
            if not dm:
                continue

            dt = _parse_date_dmy(dm.group(1))
            if not dt:
                continue

            # Month/year filter (prevents disclaimer OCR noise)
            if stmt_year_month is not None:
                yy, mm = stmt_year_month
                if dt.year != yy or dt.month != mm:
                    continue

            nums = _OCR_MONEY_RE.findall(line)
            if not nums:
                continue

            bal = _ocr_float(nums[-1])
            if bal is None:
                continue

            date_iso = dt.date().isoformat()
            candidates.setdefault(date_iso, []).append((bal, line, page_num))

    # dedupe within each date by balance value
    out: Dict[str, List[Tuple[float, str, int]]] = {}
    for d, rows in candidates.items():
        seen = set()
        uniq = []
        for bal, line, p in rows:
            key = round(float(bal), 2)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((round(float(bal), 2), line, p))
        # stable sort by page number to keep consistent description choice
        uniq.sort(key=lambda x: x[2])
        out[d] = uniq

    return out


def _resolve_one_balance_per_date(
    candidates: Dict[str, List[Tuple[float, str, int]]],
    opening_balance: float,
) -> List[Tuple[str, float, str, int]]:
    """
    February fix:
    If OCR yields multiple balances for the same date, choose the one that yields the smallest
    absolute delta vs the previous chosen balance (most plausible movement), and drop the others.
    """
    dates = sorted(candidates.keys())
    resolved: List[Tuple[str, float, str, int]] = []

    prev = float(opening_balance)

    for d in dates:
        opts = candidates[d]
        if not opts:
            continue

        if len(opts) == 1:
            bal, line, p = opts[0]
            resolved.append((d, bal, line, p))
            prev = bal
            continue

        # choose the balance with minimum |delta| from prev
        best = None  # (abs_delta, bal, line, p)
        for bal, line, p in opts:
            abs_delta = abs(round(bal - prev, 2))
            cand = (abs_delta, bal, line, p)
            if best is None or cand < best:
                best = cand

        assert best is not None
        _, bal, line, p = best
        resolved.append((d, bal, line, p))
        prev = bal

    return resolved


def _recompute_totals_from_balances(opening: float, rows: List[Tuple[str, float, str, int]]) -> Tuple[float, float]:
    prev = opening
    td = 0.0
    tc = 0.0
    for _, bal, _, _ in rows:
        delta = round(bal - prev, 2)
        if delta > 0:
            tc += delta
        elif delta < 0:
            td += abs(delta)
        prev = bal
    return round(td, 2), round(tc, 2)


def parse_bank_islam_ocr_balance_delta(pdf, source_file) -> List[Dict[str, Any]]:
    if pytesseract is None or not getattr(pdf, "pages", None):
        return []

    opening = _extract_opening_balance_via_ocr(pdf)
    if opening is None:
        return []

    # statement totals used only for validation
    stmt_td, stmt_tc = _extract_summary_totals_via_ocr(pdf)

    stmt_ym = _extract_statement_month_year_via_ocr(pdf)
    cand = _collect_date_balance_candidates_from_ocr(pdf, stmt_ym)
    if not cand:
        return []

    rows = _resolve_one_balance_per_date(cand, opening)
    if not rows:
        return []

    # Optional validation (does not mutate output unless you add further correction)
    # After February fix, this should now match the statement totals for February.
    if stmt_td is not None and stmt_tc is not None:
        td, tc = _recompute_totals_from_balances(opening, rows)
        # If still mismatching, we do NOT force aggressive changes here to avoid harming other PDFs.
        # You can add further correction logic later if needed.

    tx: List[Dict[str, Any]] = []
    prev = float(opening)

    for date_iso, bal, line, page_num in rows:
        delta = round(bal - prev, 2)
        credit = delta if delta > 0 else 0.0
        debit = abs(delta) if delta < 0 else 0.0

        # clean description
        # (remove date token if present and all money tokens from the OCR line)
        desc = line
        # remove leading date-like token
        m = _OCR_DATE_RE.match(desc)
        if m:
            desc = desc[len(m.group(1)) :].strip()

        for n in _OCR_MONEY_RE.findall(line):
            desc = desc.replace(n, "").strip()

        tx.append(
            {
                "date": date_iso,
                "description": desc,
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(bal, 2),
                "page": page_num,
                "bank": "Bank Islam",
                "source_file": source_file,
                "format": "ocr_balance_delta_v2",
            }
        )

        prev = bal

    return tx


# =========================================================
# SCANNED / GARBLED DETECTION + SUM + WRAPPER
# =========================================================
def _sum_tx(tx: List[Dict[str, Any]]) -> Tuple[float, float]:
    return (
        round(sum(t.get("debit", 0.0) or 0.0 for t in tx), 2),
        round(sum(t.get("credit", 0.0) or 0.0 for t in tx), 2),
    )


def _text_looks_garbled(txt: str) -> bool:
    if not txt:
        return True
    up = txt.upper()
    if up.count("(CID:") >= 20:
        return True
    if len(txt) > 800:
        alnum = sum(ch.isalnum() for ch in txt)
        if (alnum / max(len(txt), 1)) < 0.15:
            return True
    return False


def _looks_like_scanned(source_file: str, pdf) -> bool:
    try:
        if "scan" in (source_file or "").lower() or "scanned" in (source_file or "").lower():
            return True

        pages = getattr(pdf, "pages", []) or []
        if not pages:
            return True

        texts = []
        for p in pages[:3]:
            try:
                texts.append(((p.extract_text() or "").strip()))
            except Exception:
                texts.append("")

        for t in texts:
            if len(t) >= 120 and not _text_looks_garbled(t):
                return False

        if all((len(t) < 80) or _text_looks_garbled(t) for t in texts):
            return True

        return False
    except Exception:
        return True


def parse_bank_islam(pdf, source_file):
    tx = parse_bank_islam_format1(pdf, source_file)
    if not tx:
        tx = parse_bank_islam_format2(pdf, source_file)
    if not tx:
        tx = parse_bank_islam_format4(pdf, source_file)
    if not tx:
        tx = parse_bank_islam_format3(pdf, source_file)

    if _looks_like_scanned(source_file, pdf):
        stmt_td, stmt_tc = _extract_summary_totals_via_ocr(pdf)
        if stmt_td is not None and stmt_tc is not None:
            parsed_td, parsed_tc = _sum_tx(tx)
            if abs(parsed_td - stmt_td) > 0.01 or abs(parsed_tc - stmt_tc) > 0.01:
                tx_ocr = parse_bank_islam_ocr_balance_delta(pdf, source_file)
                if tx_ocr:
                    o_td, o_tc = _sum_tx(tx_ocr)
                    if abs(o_td - stmt_td) <= 0.01 and abs(o_tc - stmt_tc) <= 0.01:
                        return tx_ocr
                    if len(tx_ocr) > len(tx):
                        return tx_ocr

        if not tx:
            tx_ocr = parse_bank_islam_ocr_balance_delta(pdf, source_file)
            if tx_ocr:
                return tx_ocr

    return tx
