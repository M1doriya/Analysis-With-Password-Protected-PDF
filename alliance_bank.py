"""
alliance_bank.py

Parser for Alliance Bank Malaysia Berhad "Statement of Account" PDFs.

Interface (matches other bank modules in this repo):
- parse_alliance_bank(pdf: pdfplumber.PDF, filename: str) -> list[dict]
- extract_alliance_statement_totals(pdf: pdfplumber.PDF) -> dict[str, dict]
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pdfplumber

from core_utils import normalize_date, safe_float, normalize_text


# Matches money values like 1,234.56 or 1234.56
_MONEY_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*\.\d{2}|\-?\d+\.\d{2}")

# Lines that are usually page headers / column headers / footers.
_NOISE_RE_LIST = [
    re.compile(r"^\s*STATEMENT OF ACCOUNT\b", re.I),
    re.compile(r"^\s*PENYATA AKAUN\b", re.I),
    re.compile(r"^\s*PAGE\s+\d+\b", re.I),
    re.compile(r"^\s*HALAMAN\s+\d+\b", re.I),
    re.compile(r"^\s*ACCOUNT\s+NO\b", re.I),
    re.compile(r"^\s*NO\.\s*AKAUN\b", re.I),
    re.compile(r"^\s*CURRENCY\b", re.I),
    re.compile(r"^\s*MATA\s*WANG\b", re.I),
    re.compile(r"^\s*DATE\s+TARIKH\b", re.I),
    re.compile(r"^\s*TRANSACTION\s+DETAILS\b", re.I),
    re.compile(r"^\s*KETERANGAN\b", re.I),
    re.compile(r"^\s*CHEQUE\s+NO\b", re.I),
    re.compile(r"^\s*NO\.\s*CEK\b", re.I),
    re.compile(r"^\s*DEBIT\b", re.I),
    re.compile(r"^\s*KREDIT\b", re.I),
    re.compile(r"^\s*BALANCE\b", re.I),
    re.compile(r"^\s*BAKI\b", re.I),
    re.compile(r"^\s*THE ITEMS AND BALANCES SHOWN ABOVE\b", re.I),
    re.compile(r"^\s*SEGALA BUTIRAN DAN BAKI AKAUN\b", re.I),
    re.compile(r"^\s*ALLIANCE BANK MALAYSIA BERHAD\b", re.I),
    re.compile(r"^\s*WWW\.ALLIANCEBANK\.COM\.MY\b", re.I),
]


def _is_noise_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    return any(p.search(s) for p in _NOISE_RE_LIST)


def _extract_money_values(line: str) -> List[float]:
    vals = []
    for m in _MONEY_RE.findall(line or ""):
        vals.append(safe_float(m))
    return [v for v in vals if v is not None]


def _strip_trailing_cr_dr(s: str) -> str:
    # Alliance prints balance column like "54,650.98 CR" or "1,234.56 DR"
    # Keep sign marker for optional parsing later; but in most cases balance is positive in statement and CR/DR indicates side.
    s = re.sub(r"\b(CR|DR)\b\s*$", "", s.strip(), flags=re.I).strip()
    return s


def _date_token_to_iso(date_token: str) -> Optional[str]:
    # Alliance statements often use ddmmyy (e.g., 090425). We rely on core_utils.normalize_date,
    # but keep an explicit guard so we don't treat random 6-digit references as dates.
    date_token = (date_token or "").strip()
    if not re.fullmatch(r"\d{6}|\d{8}|\d{2}[/-]\d{2}[/-]\d{2,4}", date_token):
        return None
    iso = normalize_date(date_token)
    return iso


def parse_alliance_bank(pdf: pdfplumber.PDF, filename: str) -> List[dict]:
    """
    Extract transactions from Alliance statement pages.

    Typical row layout (from your samples):
      090425 <desc...> <debit?> <credit?> <balance> CR|DR
    Often debit/credit columns are blank depending on txn; we infer direction by balance delta when possible.
    """
    bank_name = "Alliance Bank"
    out: List[dict] = []

    prev_balance: Optional[float] = None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text(x_tolerance=1, y_tolerance=2) or ""
        lines = [normalize_text(x) for x in text.splitlines() if normalize_text(x)]

        for line in lines:
            if _is_noise_line(line):
                continue

            # Identify ending balance / total debit-credit lines, which are not transactions
            up = line.upper()
            if "TOTAL DEBIT/CREDIT" in up or "JUMLAH DEBIT/KREDIT" in up:
                continue

            # Detect "ENDING BALANCE" marker line (not a txn)
            if "ENDING BALANCE" in up:
                vals = _extract_money_values(line)
                if vals:
                    bal = float(vals[-1])
                    out.append(
                        {
                            "date": None,
                            "description": "ENDING BALANCE",
                            "debit": 0.0,
                            "credit": 0.0,
                            "balance": round(bal, 2),
                            "page": page_num,
                            "bank": bank_name,
                            "source_file": filename,
                            "is_balance_marker": True,
                        }
                    )
                    prev_balance = bal
                continue

            # Transaction lines: start with date-like token
            parts = line.split()
            if not parts:
                continue

            iso = _date_token_to_iso(parts[0])
            if not iso:
                continue

            money_vals = _extract_money_values(line)
            if not money_vals:
                continue

            # Heuristic: last number is balance; second last may be amount
            balance = float(money_vals[-1])
            amount = float(money_vals[-2]) if len(money_vals) >= 2 else None

            # Description: remove leading date token and remove money tokens and trailing CR/DR
            desc = line[len(parts[0]) :].strip()
            desc = _strip_trailing_cr_dr(desc)
            for m in _MONEY_RE.findall(desc):
                desc = desc.replace(m, " ")
            desc = normalize_text(desc)

            debit = 0.0
            credit = 0.0

            # Infer debit/credit:
            # If we have prev_balance, use delta. Else use presence of "DR" / "CR" or fallback to amount.
            if prev_balance is not None:
                delta = round(balance - prev_balance, 2)
                if delta > 0:
                    credit = abs(delta)
                elif delta < 0:
                    debit = abs(delta)
            else:
                # fallback using CR/DR at end of line
                tail = (line.strip().split()[-1] or "").upper()
                if amount is not None:
                    if tail == "DR":
                        debit = abs(amount)
                    elif tail == "CR":
                        credit = abs(amount)
                    else:
                        # unknown, assume credit if wording looks like inbound
                        if any(k in up for k in ["CR ADVICE", "DUITNOW CR", "CHQ DEP", "CREDIT"]):
                            credit = abs(amount)
                        else:
                            debit = abs(amount)

            prev_balance = balance

            out.append(
                {
                    "date": iso,
                    "description": desc,
                    "debit": round(float(debit), 2),
                    "credit": round(float(credit), 2),
                    "balance": round(float(balance), 2),
                    "page": page_num,
                    "bank": bank_name,
                    "source_file": filename,
                }
            )

    return out


def extract_alliance_statement_totals(pdf: pdfplumber.PDF) -> Dict[str, Dict]:
    """
    Extract statement totals per month when the statement has explicit ending balance and
    "TOTAL DEBIT/CREDIT ... <debit> <credit>" line.

    We return:
      { "YYYY-MM": { opening_balance, ending_balance, total_debit, total_credit } }

    For Alliance samples, "ENDING BALANCE" appears with date lines near end.
    Opening balance is harder (not always printed); if missing, we leave it None.
    """
    month_map: Dict[str, Dict] = {}

    full_text = "\n".join([(p.extract_text() or "") for p in pdf.pages])
    lines = [normalize_text(x) for x in full_text.splitlines() if normalize_text(x)]

    ending_balance: Optional[float] = None
    total_debit: Optional[float] = None
    total_credit: Optional[float] = None
    end_month: Optional[str] = None

    # Try to get end month from the ending balance line's date
    ending_line_re = re.compile(r"^\s*(\d{6}|\d{8}|\d{2}[/-]\d{2}[/-]\d{2,4})\s+ENDING\s+BALANCE\b", re.I)
    totals_line_re = re.compile(r"TOTAL\s+DEBIT/CREDIT|JUMLAH\s+DEBIT/KREDIT", re.I)

    for line in lines:
        if totals_line_re.search(line):
            vals = _extract_money_values(line)
            # Expect: <total_debit> <total_credit>
            if len(vals) >= 2:
                total_debit = float(vals[-2])
                total_credit = float(vals[-1])

        m_end = ending_line_re.search(line)
        if m_end:
            dt = normalize_date(m_end.group(1))
            if dt:
                end_month = dt[:7]
            vals = _extract_money_values(line)
            if vals:
                ending_balance = float(vals[-1])

    if end_month:
        month_map[end_month] = {
            "statement_month": end_month,
            "opening_balance": None,  # not reliably printed in the pages you shared
            "ending_balance": ending_balance,
            "total_debit": total_debit,
            "total_credit": total_credit,
        }

    return month_map
