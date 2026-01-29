# fraud_parser.py

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Literal

from core_utils import normalize_text, safe_float


def normalize_text_upper(text: Any) -> str:
    return normalize_text(text).upper()


def normalize_party(description: Any) -> str:
    """Normalize counterparty-ish text from a transaction description."""
    desc = normalize_text_upper(description)

    remove_patterns = [
        r"TRANSFER TO A/C",
        r"TRANSFER FR A/C",
        r"INTER-BANK PAYMENT INTO A/C",
        r"CMS\s*-\s*CR\s*PYMT",
        r"DUITNOW\s*QR-",
        r"\*",
        r"=\s*BAKI\s*LEGAR.*",
    ]
    for p in remove_patterns:
        desc = re.sub(p, " ", desc)

    desc = re.sub(r"\d{6,}", " ", desc)
    desc = re.sub(r"\s+", " ", desc).strip()

    if not desc:
        return "UNKNOWN"

    if re.fullmatch(r"[0-9 ]+", desc):
        return f"BANK_CLEARING_{desc}"

    return desc[:80]


def parse_top_parties_and_high_value(
    transactions: List[Dict[str, Any]],
    *,
    top_n: int = 5,
    high_value_threshold: float = 100_000,
    threshold_mode: Literal["gte", "lte"] = "gte",
) -> Dict[str, Any]:
    credit_by_party = defaultdict(float)
    debit_by_party = defaultdict(float)
    credit_tx_count = defaultdict(int)
    debit_tx_count = defaultdict(int)
    high_value_credits: List[Dict[str, Any]] = []

    for tx in transactions:
        party = normalize_party(tx.get("description", ""))

        credit = safe_float(tx.get("credit", 0))
        debit = safe_float(tx.get("debit", 0))

        if credit > 0:
            credit_by_party[party] += credit
            credit_tx_count[party] += 1

            is_flag = (
                (threshold_mode == "gte" and credit >= high_value_threshold)
                or (threshold_mode == "lte" and credit <= high_value_threshold)
            )
            if is_flag:
                high_value_credits.append(
                    {
                        "date": tx.get("date"),
                        "party": party,
                        "credit": round(credit, 2),
                        "description": tx.get("description"),
                        "bank": tx.get("bank"),
                        "source_file": tx.get("source_file"),
                    }
                )

        if debit > 0:
            debit_by_party[party] += debit
            debit_tx_count[party] += 1

    top_credit = sorted(credit_by_party.items(), key=lambda x: x[1], reverse=True)[: max(0, top_n)]
    top_debit = sorted(debit_by_party.items(), key=lambda x: x[1], reverse=True)[: max(0, top_n)]

    return {
        "top_credit_parties": [
            {"party": p, "total_credit": round(v, 2), "credit_tx_count": credit_tx_count[p]}
            for p, v in top_credit
        ],
        "top_debit_parties": [
            {"party": p, "total_debit": round(v, 2), "debit_tx_count": debit_tx_count[p]}
            for p, v in top_debit
        ],
        "high_value_credits": high_value_credits,
        "config": {
            "top_n": top_n,
            "high_value_threshold": high_value_threshold,
            "threshold_mode": threshold_mode,
        },
    }


def _company_tokens(company_name: str) -> List[str]:
    STOPWORDS = {
        "SDN",
        "BHD",
        "BERHAD",
        "ENTERPRISE",
        "ENT",
        "TRADING",
        "TRADERS",
        "RESOURCES",
        "COMPANY",
        "CO",
        "LTD",
        "LIMITED",
        "PERNIAGAAN",
    }

    tokens = [
        t
        for t in normalize_text_upper(company_name).split()
        if len(t) >= 3 and t not in STOPWORDS
    ]
    tokens = [t for t in tokens if not t.isdigit()]
    return tokens


def parse_inter_transactions(
    transactions: List[Dict[str, Any]],
    company_name: str,
    *,
    match_mode: Literal["any", "all", "min"] = "any",
    min_token_matches: int = 2,
) -> Dict[str, Any]:
    """Trace transactions matching a company name using word-boundary regex.

    match_mode:
      - "any": any token match flags a transaction
      - "all": all tokens must match
      - "min": at least min_token_matches tokens must match
    """
    tokens = _company_tokens(company_name)
    matched: List[Dict[str, Any]] = []

    if not tokens:
        return {
            "company_name": company_name,
            "company_tokens": tokens,
            "transaction_count": 0,
            "total_credit": 0.0,
            "total_debit": 0.0,
            "net_flow": 0.0,
            "transactions": [],
            "config": {"match_mode": match_mode, "min_token_matches": min_token_matches},
        }

    token_res = {t: re.compile(rf"\b{re.escape(t)}\b") for t in tokens}

    for tx in transactions:
        desc_norm = normalize_text_upper(tx.get("description", ""))
        party_norm = normalize_party(tx.get("description", ""))
        haystack = f"{desc_norm} {party_norm}"

        matched_tokens = [t for t, rx in token_res.items() if rx.search(haystack)]

        if match_mode == "any":
            is_match = bool(matched_tokens)
        elif match_mode == "all":
            is_match = len(matched_tokens) == len(tokens)
        else:  # "min"
            is_match = len(matched_tokens) >= max(1, int(min_token_matches))

        if is_match:
            tx_copy = dict(tx)
            tx_copy["_matched_tokens"] = matched_tokens
            matched.append(tx_copy)

    total_credit = sum(safe_float(tx.get("credit", 0)) for tx in matched)
    total_debit = sum(safe_float(tx.get("debit", 0)) for tx in matched)

    return {
        "company_name": company_name,
        "company_tokens": tokens,
        "transaction_count": len(matched),
        "total_credit": round(total_credit, 2),
        "total_debit": round(total_debit, 2),
        "net_flow": round(total_credit - total_debit, 2),
        "transactions": matched,
        "config": {"match_mode": match_mode, "min_token_matches": min_token_matches},
    }
