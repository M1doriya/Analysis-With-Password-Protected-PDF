from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None


# =========================================================
# Patterns / constants
# =========================================================

DATE_IN_TOKEN_RE = re.compile(r"(?P<d>\d{1,2})\s*/\s*(?P<m>\d{1,2})\s*/\s*(?P<y>\d{2,4})")

MONEY_TOKEN_RE = re.compile(
    r"^\(?\s*(?:RM\s*)?(?P<num>(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d{2})\s*\)?(?P<trail_sign>[+-])?\s*[\.,;:|]*\s*$",
    re.I,
)

# in text (line scanning)
MONEY_IN_TEXT_RE = re.compile(r"\d[\d,]*\.\d{2}")

HEADER_HINTS = ("DATE", "TARIKH", "DEBIT", "CREDIT", "BALANCE", "BAKI")
NON_TX_HINTS = (
    "ACCOUNT",
    "NO.",
    "STATEMENT",
    "PENYATA",
    "PAGE",
    "PIDM",
    "AFFIN",
    "BRANCH",
    "ADDRESS",
    "CUSTOMER",
    "CIF",
    "PERIOD",
    "TARIKH PENYATA",
    "STATEMENT DATE",
)
BF_HINTS = (
    "B/F",
    "BALANCE B/F",
    "BAKI B/F",
    "BAKI MULA",
    "BAKI AWAL",
    "OPENING",
    "BALANCE BROUGHT",
    "BALANCE BROUGHT FORWARD",
)

FILENAME_MONTH_RE = re.compile(r"(?P<y>20\d{2})[^\d]?(?P<m>0[1-9]|1[0-2])")


# =========================================================
# Small helpers
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _infer_month_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None
    m = FILENAME_MONTH_RE.search(filename)
    if not m:
        return None
    return f"{int(m.group('y')):04d}-{int(m.group('m')):02d}"


def _to_iso_date(token: str) -> Optional[str]:
    if not token:
        return None
    m = DATE_IN_TOKEN_RE.search(token)
    if not m:
        return None
    d = int(m.group("d"))
    mo = int(m.group("m"))
    y = int(m.group("y"))
    if y < 100:
        y += 2000
    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except Exception:
        return None


def _clean_money_token(token: str) -> Optional[str]:
    if token is None:
        return None
    t = str(token).strip()
    if not t:
        return None

    t = t.replace("O", "0").replace("o", "0").replace(" ", "")

    m = MONEY_TOKEN_RE.match(t)
    if not m:
        t2 = t.strip(".,;:|")
        m = MONEY_TOKEN_RE.match(t2)
        if not m:
            return None
        t = t2

    num = (m.group("num") or "").replace(",", "")
    if num.startswith("."):
        num = "0" + num

    sign = m.group("trail_sign")
    paren_neg = t.startswith("(") and ")" in t
    if paren_neg or sign == "-":
        return "-" + num
    return num


def _money_to_float(token: str) -> Optional[float]:
    s = _clean_money_token(token)
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _is_money_token(token: str) -> bool:
    return _clean_money_token(token) is not None


def _looks_non_tx_row(up: str) -> bool:
    return any(h in up for h in NON_TX_HINTS)


def _is_bf_row(up: str) -> bool:
    return any(h in up for h in BF_HINTS)


# =========================================================
# Totals extraction (AMENDED: robust candidate solver)
# =========================================================
def _page_text_pdf_or_ocr(page: pdfplumber.page.Page, *, crop_mode: str = "none") -> str:
    """
    Try pdf text; if empty, OCR.

    crop_mode:
      - "none": full page
      - "top": top portion (where summary tables often are)
      - "bottom": bottom portion (where totals / carried forward may be)
    """
    txt = (page.extract_text() or "").strip()
    if len(txt) >= 120:
        return txt

    if pytesseract is None:
        return txt

    try:
        p = page
        if crop_mode in ("top", "bottom"):
            w, h = float(page.width), float(page.height)
            if crop_mode == "top":
                p = page.crop((0, 0, w, h * 0.45))
            else:
                p = page.crop((0, h * 0.55, w, h))
        img = p.to_image(resolution=160).original
        return pytesseract.image_to_string(img, config="--psm 6") or ""
    except Exception:
        return txt


def _parse_money_flexible(s: str) -> Optional[float]:
    """
    Accept comma or no-comma formats: 400,620.67 or 400620.67
    """
    if not s:
        return None
    t = str(s).strip().strip(".,;:|")
    t = t.replace("O", "0").replace("o", "0").replace(" ", "")
    # keep only digits, comma, dot
    t = re.sub(r"[^0-9,\.]", "", t)
    if not t:
        return None
    t2 = t.replace(",", "")
    if not re.fullmatch(r"\d+\.\d{2}", t2):
        return None
    try:
        return float(t2)
    except Exception:
        return None


def _candidate_amounts_from_token(token: str) -> List[float]:
    """
    Generate candidate interpretations for a single OCR token.
    Critical rule: do NOT strip prefix digits when commas are missing.
    """
    if not token:
        return []
    raw = token.strip()
    cands: List[float] = []
    direct = _parse_money_flexible(raw)
    if direct is not None:
        cands.append(float(direct))

    # Only consider "count digit glued" stripping when the token has commas
    # (this is the reliable signature of that OCR defect in Affin totals lines).
    if "," in raw:
        s = raw.strip().strip(".,;:|")
        for k in (1, 2, 3):
            if len(s) <= k:
                continue
            v = _parse_money_flexible(s[k:])
            if v is not None:
                cands.append(float(v))

    # unique, stable order
    out: List[float] = []
    for x in cands:
        if x not in out:
            out.append(x)
    return out


def _money_tokens_in_line(line: str) -> List[str]:
    return [x.strip() for x in MONEY_IN_TEXT_RE.findall(line or "")]


def _scan_lines_for_totals_candidates(text: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {
        "opening_balance": [],
        "total_debit": [],
        "total_credit": [],
        "ending_balance": [],
    }
    if not text:
        return out

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        up = line.upper()

        toks = _money_tokens_in_line(line)
        if not toks:
            continue

        # Prefer last token but also consider others (OCR sometimes shifts)
        # We will add candidates for each token (lightweight; solver will pick consistent set).
        token_pool = toks[-2:] if len(toks) >= 2 else toks

        def add(key: str):
            for tok in token_pool:
                out[key].extend(_candidate_amounts_from_token(tok))

        # Total debit
        hit_debit = (
            ((("TOTAL" in up or "JUMLAH" in up) and "DEBIT" in up))
            or ("JUMLAH" in up and ("PENGELUARAN" in up or "KELUAR" in up))
        )
        if hit_debit:
            add("total_debit")
            continue

        # Total credit
        hit_credit = (
            ((("TOTAL" in up or "JUMLAH" in up) and ("CREDIT" in up or "KREDIT" in up)))
            or ("JUMLAH" in up and ("PEMASUKAN" in up or "MASUK" in up))
        )
        if hit_credit:
            add("total_credit")
            continue

        # Opening / B/F
        hit_open = (
            ("B/F" in up)
            or ("BROUGHT FORWARD" in up)
            or ("OPENING" in up and "BAL" in up)
            or ("BAKI" in up and ("AWAL" in up or "MULA" in up))
        )
        if hit_open:
            add("opening_balance")
            continue

        # Ending / C/F
        hit_end = (
            ("C/F" in up)
            or ("CARRIED FORWARD" in up)
            or ("ENDING BALANCE" in up)
            or ("CLOSING BALANCE" in up)
            or ("BAKI" in up and ("AKHIR" in up or "PENUTUP" in up or "TUTUP" in up))
        )
        if hit_end:
            add("ending_balance")
            continue

    # unique each list
    for k in out:
        uniq: List[float] = []
        for v in out[k]:
            if v not in uniq:
                uniq.append(v)
        out[k] = uniq

    return out


def _choose_best_totals(cands: Dict[str, List[float]]) -> Dict[str, Optional[float]]:
    """
    Choose totals using accounting identity:
      opening + credit - debit ~= ending

    Also penalize suspicious “tiny” values when others are large (strip artifact).
    """
    opens = sorted(cands.get("opening_balance", []), reverse=True)
    ends = sorted(cands.get("ending_balance", []), reverse=True)
    debits = sorted(cands.get("total_debit", []), reverse=True)
    credits = sorted(cands.get("total_credit", []), reverse=True)

    best = {"opening_balance": None, "total_debit": None, "total_credit": None, "ending_balance": None}

    # nothing to solve
    if not opens and not ends and not debits and not credits:
        return best

    def penalty(o: float, d: float, c: float, e: float) -> float:
        # Penalize “620.67” type artifacts when scale is clearly higher
        vals = [o, d, c, e]
        mx = max(vals)
        mn = min(vals)
        p = 0.0
        if mx >= 10000 and mn < 1000:
            p += 1000.0
        return p

    tol = 0.06
    best_score: Optional[float] = None

    # Search top candidates only (performance + stability)
    for o in (opens[:10] or [None]):
        for e in (ends[:10] or [None]):
            for d in (debits[:15] or [None]):
                for c in (credits[:15] or [None]):
                    if o is None or e is None or d is None or c is None:
                        continue
                    err = abs((o + c - d) - e)
                    score = err + penalty(o, d, c, e)
                    if err <= tol:
                        if best_score is None or score < best_score:
                            best_score = score
                            best = {"opening_balance": o, "total_debit": d, "total_credit": c, "ending_balance": e}

    # If exact-ish solution not found, pick “most plausible” by minimizing error anyway
    if best["opening_balance"] is None:
        for o in (opens[:10] or [None]):
            for e in (ends[:10] or [None]):
                for d in (debits[:15] or [None]):
                    for c in (credits[:15] or [None]):
                        if o is None or e is None or d is None or c is None:
                            continue
                        err = abs((o + c - d) - e)
                        score = err + penalty(o, d, c, e)
                        if best_score is None or score < best_score:
                            best_score = score
                            best = {"opening_balance": o, "total_debit": d, "total_credit": c, "ending_balance": e}

    # Fallbacks if still partial
    if best["opening_balance"] is None and opens:
        best["opening_balance"] = opens[0]
    if best["ending_balance"] is None and ends:
        best["ending_balance"] = ends[0]
    if best["total_debit"] is None and debits:
        best["total_debit"] = debits[0]
    if best["total_credit"] is None and credits:
        best["total_credit"] = credits[0]

    return best


def extract_affin_statement_totals(pdf_input: Any, source_file: str = "") -> Dict[str, Any]:
    """
    Source of truth for monthly summary:
      opening_balance, total_debit, total_credit, ending_balance

    This function is designed to be robust even when OCR:
      - drops commas
      - injects count digits
      - misreads one token (solver uses accounting identity)
    """
    if hasattr(pdf_input, "pages") and hasattr(pdf_input, "close"):
        pdf = pdf_input
        should_close = False
    else:
        should_close = True
        if isinstance(pdf_input, (bytes, bytearray)):
            pdf = pdfplumber.open(BytesIO(bytes(pdf_input)))
        elif hasattr(pdf_input, "getvalue"):
            pdf = pdfplumber.open(BytesIO(pdf_input.getvalue()))
        else:
            pdf = pdfplumber.open(pdf_input)

    try:
        n = len(pdf.pages)
        idxs: List[int] = []
        for i in [0, 1, max(0, n - 2), max(0, n - 1)]:
            if 0 <= i < n and i not in idxs:
                idxs.append(i)

        merged = {"opening_balance": [], "total_debit": [], "total_credit": [], "ending_balance": []}

        for i in idxs:
            p = pdf.pages[i]
            # Scan multiple views of the page: full + top + bottom
            for mode in ("none", "top", "bottom"):
                text = _page_text_pdf_or_ocr(p, crop_mode=mode).replace("\x0c", " ")
                found = _scan_lines_for_totals_candidates(text)
                for k in merged:
                    merged[k].extend(found.get(k, []))

        # unique merged
        for k in merged:
            uniq: List[float] = []
            for v in merged[k]:
                if v not in uniq:
                    uniq.append(v)
            merged[k] = uniq

        best = _choose_best_totals(merged)

        return {
            "bank": "Affin Bank",
            "source_file": source_file or "",
            "statement_month": _infer_month_from_filename(source_file),
            "opening_balance": best["opening_balance"],
            "total_debit": best["total_debit"],
            "total_credit": best["total_credit"],
            "ending_balance": best["ending_balance"],
        }
    finally:
        if should_close:
            try:
                pdf.close()
            except Exception:
                pass


# =========================================================
# Transaction extraction (unchanged; not used for monthly totals)
# =========================================================
def _words_from_pdf(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    words = page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
        extra_attrs=["x0", "x1", "top", "bottom"],
    ) or []
    out: List[Dict[str, Any]] = []
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        out.append(
            {
                "text": t,
                "x0": float(w.get("x0", 0.0)),
                "x1": float(w.get("x1", 0.0)),
                "y0": float(w.get("top", 0.0)),
                "y1": float(w.get("bottom", 0.0)),
            }
        )
    return out


def _words_from_ocr(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    if pytesseract is None:
        return []
    try:
        w, h = float(page.width), float(page.height)
        crop = page.crop((0, 80, w, h - 50))
        img = crop.to_image(resolution=240).original
    except Exception:
        img = page.to_image(resolution=240).original

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 6")
    n = len(data.get("text", []))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x, y, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append({"text": txt, "x0": float(x), "x1": float(x + ww), "y0": float(y), "y1": float(y + hh)})
    return out


def _get_page_words(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    w = _words_from_pdf(page)
    if w:
        return w
    return _words_from_ocr(page)


def _cluster_rows(words: List[Dict[str, Any]], y_tol: float = 2.8) -> List[Tuple[float, List[Dict[str, Any]]]]:
    if not words:
        return []
    words.sort(key=lambda r: (r["y0"], r["x0"]))
    buckets: List[Dict[str, Any]] = []
    for w in words:
        placed = False
        for b in buckets:
            if abs(w["y0"] - b["y"]) <= y_tol:
                b["items"].append(w)
                b["y"] = (b["y"] * (len(b["items"]) - 1) + w["y0"]) / len(b["items"])
                placed = True
                break
        if not placed:
            buckets.append({"y": w["y0"], "items": [w]})
    out: List[Tuple[float, List[Dict[str, Any]]]] = []
    for b in sorted(buckets, key=lambda z: z["y"]):
        out.append((float(b["y"]), sorted(b["items"], key=lambda z: z["x0"])))
    return out


def _row_text(row_words: List[Dict[str, Any]]) -> str:
    return _norm(" ".join(w["text"] for w in row_words))


def _row_has_date(row_words: List[Dict[str, Any]]) -> bool:
    for w in row_words[:10]:
        if _to_iso_date(w["text"]):
            return True
    return False


def _detect_columns(rows: List[Tuple[float, List[Dict[str, Any]]]]) -> Optional[Dict[str, float]]:
    for _, rw in rows[:80]:
        up = _row_text(rw).upper()
        if not any(h in up for h in HEADER_HINTS):
            continue
        debit_x = credit_x = balance_x = None
        for w in rw:
            t = w["text"].upper()
            xc = (w["x0"] + w["x1"]) / 2.0
            if debit_x is None and ("DEBIT" in t or t == "DR"):
                debit_x = xc
            if credit_x is None and ("CREDIT" in t or "CR" in t):
                credit_x = xc
            if balance_x is None and ("BAL" in t or "BAKI" in t):
                balance_x = xc
        if debit_x and credit_x and balance_x:
            return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}
    return None


def _classify_money_by_columns(
    row_words: List[Dict[str, Any]], col: Optional[Dict[str, float]]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    money_items: List[Tuple[float, float]] = []
    for w in row_words:
        if not _is_money_token(w["text"]):
            continue
        v = _money_to_float(w["text"])
        if v is None:
            continue
        xc = float((w["x0"] + w["x1"]) / 2.0)
        money_items.append((xc, float(v)))
    if not money_items:
        return None, None, None

    money_items.sort(key=lambda t: t[0])
    if not col:
        return None, None, money_items[-1][1]

    debit_x = float(col.get("debit_x", -1))
    credit_x = float(col.get("credit_x", -1))
    balance_x = float(col.get("balance_x", -1))

    debit_vals: List[float] = []
    credit_vals: List[float] = []
    balance_vals: List[float] = []

    for xc, v in money_items:
        candidates = []
        if debit_x > 0:
            candidates.append(("debit", abs(xc - debit_x)))
        if credit_x > 0:
            candidates.append(("credit", abs(xc - credit_x)))
        if balance_x > 0:
            candidates.append(("balance", abs(xc - balance_x)))
        label, dist = min(candidates, key=lambda x: x[1])
        if dist > 90:
            continue
        if label == "debit":
            debit_vals.append(abs(v))
        elif label == "credit":
            credit_vals.append(abs(v))
        else:
            balance_vals.append(v)

    debit = round(sum(debit_vals), 2) if debit_vals else None
    credit = round(sum(credit_vals), 2) if credit_vals else None
    balance = balance_vals[-1] if balance_vals else money_items[-1][1]
    return debit, credit, float(balance)


def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    bank_name = "Affin Bank"
    txs: List[Dict[str, Any]] = []

    if hasattr(pdf_input, "pages") and hasattr(pdf_input, "close"):
        pdf = pdf_input
        should_close = False
    else:
        should_close = True
        if isinstance(pdf_input, (bytes, bytearray)):
            pdf = pdfplumber.open(BytesIO(bytes(pdf_input)))
        elif hasattr(pdf_input, "getvalue"):
            pdf = pdfplumber.open(BytesIO(pdf_input.getvalue()))
        else:
            pdf = pdfplumber.open(pdf_input)

    try:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = _get_page_words(page)
            if not words:
                continue
            rows = _cluster_rows(words, y_tol=2.8)
            if not rows:
                continue
            col = _detect_columns(rows)

            i = 0
            while i < len(rows):
                row_y, row_words = rows[i]
                txt = _row_text(row_words)
                if not txt:
                    i += 1
                    continue
                up = txt.upper()

                if _looks_non_tx_row(up) and not _row_has_date(row_words):
                    i += 1
                    continue

                date_iso = None
                for w in row_words[:10]:
                    d = _to_iso_date(w["text"])
                    if d:
                        date_iso = d
                        break
                if not date_iso:
                    i += 1
                    continue

                # merge wrapped lines
                block_words = list(row_words)
                k = i + 1
                while k < len(rows) and not _row_has_date(rows[k][1]):
                    nxt_up = _row_text(rows[k][1]).upper()
                    if any(h in nxt_up for h in HEADER_HINTS):
                        break
                    block_words.extend(rows[k][1])
                    k += 1
                block_words.sort(key=lambda z: (z["y0"], z["x0"]))

                debit, credit, balance = _classify_money_by_columns(block_words, col)
                if balance is None:
                    i = k
                    continue

                if _is_bf_row(up):
                    i = k
                    continue

                desc_parts: List[str] = []
                for ww in block_words:
                    t = (ww.get("text") or "").strip().strip("|")
                    if not t:
                        continue
                    if _is_money_token(t):
                        continue
                    if _to_iso_date(t) == date_iso:
                        continue
                    desc_parts.append(t)

                description = _norm(" ".join(desc_parts))

                txs.append(
                    {
                        "date": date_iso,
                        "description": description,
                        "debit": round(float(debit or 0.0), 2),
                        "credit": round(float(credit or 0.0), 2),
                        "balance": round(float(balance), 2),
                        "page": int(page_num),
                        "bank": bank_name,
                        "source_file": source_file or "",
                        "_y": float(row_y),
                    }
                )
                i = k

    finally:
        if should_close:
            try:
                pdf.close()
            except Exception:
                pass

    txs.sort(key=lambda x: (x.get("date", ""), int(x.get("page") or 0), float(x.get("_y") or 0.0)))
    for t in txs:
        t.pop("_y", None)
    return txs
