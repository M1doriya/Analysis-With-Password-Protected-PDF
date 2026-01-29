"""
app.py

Streamlit app: Upload bank statement PDF(s) -> parse transactions -> output standardized JSON + monthly summary.

Key upgrades in this version:
- Added Alliance Bank parser
- Added password input + decryption for encrypted PDFs
- Added mandatory OCR fallback when no transactions are extracted
- Standardized monthly summary fields: month, transaction_count, opening_balance,
  total_debit, total_credit, net_change, highest_balance, lowest_balance, source_files
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Callable, Dict, List, Optional

import pandas as pd
import pdfplumber
import streamlit as st

from core_utils import (
    dedupe_transactions,
    dedupe_transactions_affin,
    normalize_transactions,
)

from pdf_security import decrypt_pdf_bytes, is_pdf_encrypted
from ocr_fallback import ocr_pdf_to_text_pages, parse_transactions_from_ocr_text_pages

# Existing bank parsers (already in your repo)
from maybank import parse_transactions_maybank
from public_bank import parse_transactions_pbb
from rhb import parse_transactions_rhb
from cimb import parse_transactions_cimb, extract_cimb_statement_totals
from bank_islam import parse_bank_islam
from bank_rakyat import parse_bank_rakyat
from hong_leong import parse_hong_leong
from ambank import parse_ambank, extract_ambank_statement_totals
from bank_muamalat import parse_transactions_bank_muamalat
from affin_bank import parse_affin_bank, extract_affin_statement_totals
from agro_bank import parse_agro_bank
from ocbc import parse_transactions_ocbc

# New bank: Alliance
from alliance_bank import parse_alliance_bank, extract_alliance_statement_totals


def _parse_with_pdfplumber(parser_func: Callable, pdf_bytes: bytes, filename: str) -> List[dict]:
    """
    Open PDF bytes using pdfplumber and call bank parser.

    We keep this wrapper so individual bank modules don't have to worry about file streams.
    """
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        # Most bank parser funcs in this repo have signature (pdf, filename)
        return parser_func(pdf, filename)


def _extract_statement_totals(bank_choice: str, pdf_bytes: bytes) -> Dict[str, Dict]:
    """
    Optional: parse statement-level totals (opening/ending/total debit/credit) for banks that support it.
    Output structure: { "YYYY-MM": { ... } }
    """
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        if bank_choice == "Affin Bank":
            return extract_affin_statement_totals(pdf)
        if bank_choice == "AmBank":
            return extract_ambank_statement_totals(pdf)
        if bank_choice == "CIMB Bank":
            return extract_cimb_statement_totals(pdf)
        if bank_choice == "Alliance Bank":
            return extract_alliance_statement_totals(pdf)
    return {}


def _count_real_transactions(txns: List[dict]) -> int:
    """
    Exclude balance-marker rows from 'transaction_count'.
    """
    n = 0
    for t in txns:
        desc = (t.get("description") or "").upper()
        is_marker = bool(t.get("is_balance_marker")) or (
            "BEGINNING BALANCE" in desc
            or "ENDING BALANCE" in desc
            or "OPENING BALANCE" in desc
            or "BALANCE B/F" in desc
            or "BALANCE BROUGHT FORWARD" in desc
        )
        if is_marker:
            continue
        if float(t.get("debit") or 0) != 0 or float(t.get("credit") or 0) != 0:
            n += 1
    return n


def _build_monthly_summary(df: pd.DataFrame, statement_totals_by_file: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    """
    Standard monthly summary table with required columns:
      month, transaction_count, opening_balance, total_debit, total_credit,
      net_change, highest_balance, lowest_balance, source_files

    We compute from normalized transactions, then (optionally) overlay statement totals if available.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "transaction_count",
                "opening_balance",
                "total_debit",
                "total_credit",
                "net_change",
                "highest_balance",
                "lowest_balance",
                "source_files",
            ]
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.strftime("%Y-%m")

    # numeric columns
    for c in ["debit", "credit", "balance"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # group monthly
    groups = []
    for month, g in df.groupby("month", dropna=True):
        g_sorted = g.sort_values(["date", "page"], ascending=[True, True], na_position="last")
        tx_count = 0
        for _, row in g_sorted.iterrows():
            desc = str(row.get("description") or "").upper()
            if (
                "BEGINNING BALANCE" in desc
                or "OPENING BALANCE" in desc
                or "BALANCE B/F" in desc
                or "BALANCE BROUGHT FORWARD" in desc
                or row.get("is_balance_marker")
            ):
                continue
            if float(row.get("debit") or 0) != 0 or float(row.get("credit") or 0) != 0:
                tx_count += 1

        opening_balance = None
        ending_balance = None
        balances = g_sorted["balance"].dropna().tolist()

        if balances:
            # opening = first non-null balance in month
            opening_balance = float(balances[0])
            ending_balance = float(balances[-1])

        total_debit = float(g_sorted["debit"].fillna(0).sum())
        total_credit = float(g_sorted["credit"].fillna(0).sum())

        net_change = None
        if opening_balance is not None and ending_balance is not None:
            net_change = float(ending_balance - opening_balance)
        elif total_debit or total_credit:
            net_change = float(total_credit - total_debit)

        highest_balance = float(max(balances)) if balances else None
        lowest_balance = float(min(balances)) if balances else None

        source_files = sorted(set([str(x) for x in g_sorted["source_file"].dropna().tolist()]))

        groups.append(
            {
                "month": month,
                "transaction_count": int(tx_count),
                "opening_balance": None if opening_balance is None else round(opening_balance, 2),
                "total_debit": round(total_debit, 2),
                "total_credit": round(total_credit, 2),
                "net_change": None if net_change is None else round(net_change, 2),
                "highest_balance": None if highest_balance is None else round(highest_balance, 2),
                "lowest_balance": None if lowest_balance is None else round(lowest_balance, 2),
                "source_files": source_files,
            }
        )

    summary = pd.DataFrame(groups).sort_values("month")

    # Optional overlay from statement totals:
    # statement_totals_by_file: filename -> { month -> {opening_balance, ending_balance, total_debit, total_credit, ...}}
    # We will fill missing opening/ending or override if statement values exist.
    for fname, month_map in (statement_totals_by_file or {}).items():
        for month, totals in (month_map or {}).items():
            if summary.empty:
                continue
            idx = summary.index[summary["month"] == month].tolist()
            if not idx:
                continue
            i = idx[0]

            ob = totals.get("opening_balance")
            eb = totals.get("ending_balance")
            td = totals.get("total_debit")
            tc = totals.get("total_credit")

            # prefer statement opening/ending when available (more reliable than inferred from balances)
            if ob is not None:
                summary.at[i, "opening_balance"] = round(float(ob), 2)
            if eb is not None:
                # we don't have ending_balance column; reflect into net_change by recompute below
                ending_balance = float(eb)
                opening_balance = summary.at[i, "opening_balance"]
                if opening_balance is not None and pd.notna(opening_balance):
                    summary.at[i, "net_change"] = round(ending_balance - float(opening_balance), 2)

            # statement totals override computed totals (keeps consistent with statement)
            if td is not None:
                summary.at[i, "total_debit"] = round(float(td), 2)
            if tc is not None:
                summary.at[i, "total_credit"] = round(float(tc), 2)

            # ensure source_files includes this file
            sf = summary.at[i, "source_files"]
            if isinstance(sf, list) and fname not in sf:
                sf.append(fname)
                summary.at[i, "source_files"] = sorted(sf)

    # recompute net_change from totals if missing
    for i in range(len(summary)):
        if pd.isna(summary.at[i, "net_change"]) or summary.at[i, "net_change"] is None:
            summary.at[i, "net_change"] = round(float(summary.at[i, "total_credit"]) - float(summary.at[i, "total_debit"]), 2)

    return summary


def main() -> None:
    st.set_page_config(page_title="Bank Statement Parser", page_icon="🏦", layout="wide")
    st.title("🏦 Bank Statement Parser")

    # --- Session state init ---
    for key, default in [
        ("results", []),
        ("errors", []),
        ("statement_totals_by_file", {}),  # filename -> {month->totals}
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    bank_choice = st.selectbox(
        "Select bank",
        [
            "Maybank",
            "Public Bank",
            "RHB Bank",
            "CIMB Bank",
            "Bank Islam",
            "Bank Rakyat",
            "Hong Leong Bank",
            "AmBank",
            "Bank Muamalat",
            "Affin Bank",
            "Agrobank",
            "OCBC",
            "Alliance Bank",
        ],
    )

    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    # Bank parser mapping — normalized to signature (pdf_bytes, filename) -> txns_raw
    bank_parsers: Dict[str, Callable[[bytes, str], List[dict]]] = {
        "Maybank": lambda b, f: parse_transactions_maybank(b, f),
        "Public Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_pbb, b, f),
        "RHB Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_rhb, b, f),
        "CIMB Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_cimb, b, f),
        "Bank Islam": lambda b, f: _parse_with_pdfplumber(parse_bank_islam, b, f),
        "Bank Rakyat": lambda b, f: _parse_with_pdfplumber(parse_bank_rakyat, b, f),
        "Hong Leong Bank": lambda b, f: _parse_with_pdfplumber(parse_hong_leong, b, f),
        "AmBank": lambda b, f: _parse_with_pdfplumber(parse_ambank, b, f),
        "Bank Muamalat": lambda b, f: _parse_with_pdfplumber(parse_transactions_bank_muamalat, b, f),
        "Affin Bank": lambda b, f: _parse_with_pdfplumber(parse_affin_bank, b, f),
        "Agrobank": lambda b, f: _parse_with_pdfplumber(parse_agro_bank, b, f),
        "OCBC": lambda b, f: _parse_with_pdfplumber(parse_transactions_ocbc, b, f),
        "Alliance Bank": lambda b, f: _parse_with_pdfplumber(parse_alliance_bank, b, f),
    }

    # Detect encrypted files to trigger password UI
    encrypted_names: List[str] = []
    if uploaded_files:
        for uf in uploaded_files:
            try:
                if is_pdf_encrypted(uf.getvalue()):
                    encrypted_names.append(uf.name)
            except Exception:
                # If detection fails, we assume it might be encrypted.
                encrypted_names.append(uf.name)

    pdf_password: Optional[str] = None
    if encrypted_names:
        st.warning(
            "One or more PDFs look password-protected. Enter the password once and it will be applied to all encrypted files:\n\n"
            + "\n".join([f"- {n}" for n in encrypted_names])
        )
        pdf_password = st.text_input("PDF password", type="password", value="")

    force_ocr = st.checkbox("Force OCR (run OCR even if text extraction works)", value=False)

    if st.button("Start Processing", type="primary", disabled=not uploaded_files):
        st.session_state.results = []
        st.session_state.errors = []
        st.session_state.statement_totals_by_file = {}

        progress = st.progress(0)
        for i, uf in enumerate(uploaded_files, start=1):
            fname = uf.name
            try:
                raw_bytes = uf.getvalue()

                # Decrypt if needed
                if is_pdf_encrypted(raw_bytes):
                    raw_bytes = decrypt_pdf_bytes(raw_bytes, pdf_password)

                # Parse statement totals (optional, used to fill missing summary fields)
                totals = _extract_statement_totals(bank_choice, raw_bytes)
                if totals:
                    st.session_state.statement_totals_by_file[fname] = totals

                # Parse using bank-specific parser
                txns_raw = bank_parsers[bank_choice](raw_bytes, fname) or []

                txns = normalize_transactions(txns_raw, default_bank=bank_choice, source_file=fname)
                if bank_choice == "Affin Bank":
                    txns = dedupe_transactions_affin(txns)
                else:
                    txns = dedupe_transactions(txns)

                # Mandatory OCR fallback: if no real txns found OR force_ocr checked
                if force_ocr or _count_real_transactions(txns) == 0:
                    st.info(f"OCR fallback triggered for: {fname}")
                    pages_text = ocr_pdf_to_text_pages(raw_bytes, dpi=220)
                    ocr_raw = parse_transactions_from_ocr_text_pages(pages_text, bank_choice, fname)

                    ocr_txns = normalize_transactions(ocr_raw, default_bank=bank_choice, source_file=fname)
                    if bank_choice == "Affin Bank":
                        merged = dedupe_transactions_affin(txns + ocr_txns)
                    else:
                        merged = dedupe_transactions(txns + ocr_txns)

                    txns = merged

                st.session_state.results.extend(txns)

            except Exception as e:
                st.session_state.errors.append({"file": fname, "error": str(e)})
            finally:
                progress.progress(i / max(1, len(uploaded_files)))

        progress.empty()
        st.success("Processing completed.")

    # --- Results ---
    if st.session_state.errors:
        st.error("Some files failed to process:")
        st.json(st.session_state.errors)

    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)

        st.subheader("Extracted Transactions")
        st.dataframe(df, use_container_width=True)

        summary_df = _build_monthly_summary(df, st.session_state.statement_totals_by_file)
        st.subheader("Monthly Summary")
        st.dataframe(summary_df, use_container_width=True)

        output = {
            "bank": bank_choice,
            "transactions": st.session_state.results,
            "monthly_summary": summary_df.to_dict(orient="records"),
        }

        st.download_button(
            "Download Combined JSON",
            data=json.dumps(output, indent=2, ensure_ascii=False),
            file_name="bank_statement_output.json",
            mime="application/json",
        )

        st.download_button(
            "Download Transactions JSON",
            data=json.dumps(st.session_state.results, indent=2, ensure_ascii=False),
            file_name="transactions.json",
            mime="application/json",
        )

        st.download_button(
            "Download Monthly Summary JSON",
            data=json.dumps(summary_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
            file_name="monthly_summary.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
