"""
app.py (LAZY-IMPORT SAFE VERSION)

Fixes recurring ImportError crashes by:
- NOT importing every bank module at startup
- NOT importing OCR dependencies at startup
- Importing only the selected bank module when processing
- Catching and displaying the real error in the UI (no more redacted guessing)
"""

from __future__ import annotations

import importlib
import json
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber
import streamlit as st

from core_utils import dedupe_transactions, dedupe_transactions_affin, normalize_transactions

# Keep these imports (small + should be present). If you want ultra-safe mode,
# you can lazy-import pdf_security too, but start with this.
from pdf_security import decrypt_pdf_bytes, is_pdf_encrypted


# -----------------------------------------------------------------------------
# Bank registry: declare module + candidate function names
# -----------------------------------------------------------------------------
# mode:
#   - "bytes": parser signature likely (pdf_bytes, filename)
#   - "pdf":   parser signature likely (pdfplumber_pdf, filename)
BANK_SPECS: Dict[str, Dict[str, Any]] = {
    "Maybank": {
        "module": "maybank",
        "mode": "bytes",
        "parse_candidates": ["parse_transactions_maybank", "parse_maybank", "parse_transactions"],
        "totals_candidates": [],
    },
    "Public Bank": {
        "module": "public_bank",
        "mode": "pdf",
        "parse_candidates": ["parse_transactions_pbb", "parse_public_bank", "parse_transactions"],
        "totals_candidates": [],
    },
    "RHB Bank": {
        "module": "rhb",
        "mode": "pdf",
        "parse_candidates": ["parse_transactions_rhb", "parse_rhb", "parse_transactions"],
        "totals_candidates": [],
    },
    "CIMB Bank": {
        "module": "cimb",
        "mode": "pdf",
        "parse_candidates": ["parse_transactions_cimb", "parse_cimb", "parse_transactions"],
        "totals_candidates": ["extract_cimb_statement_totals"],
    },
    "Bank Islam": {
        "module": "bank_islam",
        "mode": "pdf",
        "parse_candidates": ["parse_bank_islam", "parse_transactions_bank_islam", "parse_transactions"],
        "totals_candidates": [],
    },
    "Bank Rakyat": {
        "module": "bank_rakyat",
        "mode": "pdf",
        "parse_candidates": ["parse_bank_rakyat", "parse_transactions_bank_rakyat", "parse_transactions"],
        "totals_candidates": [],
    },
    "Hong Leong Bank": {
        "module": "hong_leong",
        "mode": "pdf",
        "parse_candidates": ["parse_hong_leong", "parse_transactions_hong_leong", "parse_transactions"],
        "totals_candidates": [],
    },
    "AmBank": {
        "module": "ambank",
        "mode": "pdf",
        "parse_candidates": ["parse_ambank", "parse_transactions_ambank", "parse_transactions"],
        "totals_candidates": ["extract_ambank_statement_totals"],
    },
    "Bank Muamalat": {
        "module": "bank_muamalat",
        "mode": "pdf",
        "parse_candidates": ["parse_transactions_bank_muamalat", "parse_bank_muamalat", "parse_transactions"],
        "totals_candidates": [],
    },
    "Affin Bank": {
        "module": "affin_bank",
        "mode": "pdf",
        "parse_candidates": ["parse_affin_bank", "parse_transactions_affin", "parse_transactions"],
        "totals_candidates": ["extract_affin_statement_totals"],
    },
    "Agrobank": {
        "module": "agro_bank",
        "mode": "pdf",
        "parse_candidates": ["parse_agro_bank", "parse_transactions_agrobank", "parse_transactions"],
        "totals_candidates": [],
    },
    "OCBC": {
        "module": "ocbc",
        "mode": "pdf",
        "parse_candidates": ["parse_transactions_ocbc", "parse_ocbc", "parse_transactions"],
        "totals_candidates": [],
    },
    "Alliance Bank": {
        "module": "alliance_bank",
        "mode": "pdf",
        "parse_candidates": ["parse_alliance_bank", "parse_transactions_alliance", "parse_transactions"],
        "totals_candidates": ["extract_alliance_statement_totals"],
    },
}


def _pick_attr(mod, candidates: List[str]) -> Callable:
    for name in candidates:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            if callable(fn):
                return fn
    raise ImportError(f"Module '{mod.__name__}' does not have any of: {candidates}")


def _optional_attr(mod, candidates: List[str]) -> Optional[Callable]:
    for name in candidates:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            if callable(fn):
                return fn
    return None


def _load_bank(bank_choice: str) -> Tuple[Callable, Optional[Callable], str]:
    """
    Returns: (parse_fn, totals_fn_or_none, mode)
    """
    spec = BANK_SPECS[bank_choice]
    mod = importlib.import_module(spec["module"])
    parse_fn = _pick_attr(mod, spec["parse_candidates"])
    totals_fn = _optional_attr(mod, spec.get("totals_candidates", []))
    return parse_fn, totals_fn, spec["mode"]


def _parse_transactions(parse_fn: Callable, mode: str, pdf_bytes: bytes, filename: str) -> List[dict]:
    if mode == "bytes":
        return parse_fn(pdf_bytes, filename) or []
    # mode == "pdf"
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        return parse_fn(pdf, filename) or []


def _extract_statement_totals(totals_fn: Optional[Callable], pdf_bytes: bytes) -> Dict[str, Dict]:
    if not totals_fn:
        return {}
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        out = totals_fn(pdf)
    return out or {}


def _count_real_transactions(txns: List[dict]) -> int:
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

    for c in ["debit", "credit", "balance"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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

        balances = g_sorted["balance"].dropna().tolist()
        opening_balance = float(balances[0]) if balances else None
        ending_balance = float(balances[-1]) if balances else None

        total_debit = float(g_sorted["debit"].fillna(0).sum())
        total_credit = float(g_sorted["credit"].fillna(0).sum())

        if opening_balance is not None and ending_balance is not None:
            net_change = float(ending_balance - opening_balance)
        else:
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

    # overlay statement totals when provided
    for fname, month_map in (statement_totals_by_file or {}).items():
        for month, totals in (month_map or {}).items():
            idx = summary.index[summary["month"] == month].tolist()
            if not idx:
                continue
            i = idx[0]
            ob = totals.get("opening_balance")
            eb = totals.get("ending_balance")
            td = totals.get("total_debit")
            tc = totals.get("total_credit")

            if ob is not None:
                summary.at[i, "opening_balance"] = round(float(ob), 2)
            if td is not None:
                summary.at[i, "total_debit"] = round(float(td), 2)
            if tc is not None:
                summary.at[i, "total_credit"] = round(float(tc), 2)
            if eb is not None:
                ending_balance = float(eb)
                opening_balance = summary.at[i, "opening_balance"]
                if opening_balance is not None and pd.notna(opening_balance):
                    summary.at[i, "net_change"] = round(ending_balance - float(opening_balance), 2)

            sf = summary.at[i, "source_files"]
            if isinstance(sf, list) and fname not in sf:
                sf.append(fname)
                summary.at[i, "source_files"] = sorted(sf)

    # fill missing net_change
    for i in range(len(summary)):
        if pd.isna(summary.at[i, "net_change"]) or summary.at[i, "net_change"] is None:
            summary.at[i, "net_change"] = round(float(summary.at[i, "total_credit"]) - float(summary.at[i, "total_debit"]), 2)

    return summary


def main() -> None:
    st.set_page_config(page_title="Bank Statement Parser", page_icon="🏦", layout="wide")
    st.title("🏦 Bank Statement Parser")

    if "results" not in st.session_state:
        st.session_state.results = []
    if "errors" not in st.session_state:
        st.session_state.errors = []
    if "statement_totals_by_file" not in st.session_state:
        st.session_state.statement_totals_by_file = {}
    if "last_exception" not in st.session_state:
        st.session_state.last_exception = None

    bank_choice = st.selectbox("Select bank", list(BANK_SPECS.keys()))
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    # Detect encrypted files (safe)
    encrypted_names: List[str] = []
    if uploaded_files:
        for uf in uploaded_files:
            try:
                if is_pdf_encrypted(uf.getvalue()):
                    encrypted_names.append(uf.name)
            except Exception:
                encrypted_names.append(uf.name)

    pdf_password: Optional[str] = None
    if encrypted_names:
        st.warning(
            "Encrypted PDFs detected. Enter password once for all encrypted files:\n\n"
            + "\n".join([f"- {n}" for n in encrypted_names])
        )
        pdf_password = st.text_input("PDF password", type="password", value="")

    force_ocr = st.checkbox("Force OCR (run OCR even if text extraction works)", value=False)

    if st.button("Start Processing", type="primary", disabled=not uploaded_files):
        st.session_state.results = []
        st.session_state.errors = []
        st.session_state.statement_totals_by_file = {}
        st.session_state.last_exception = None

        try:
            parse_fn, totals_fn, mode = _load_bank(bank_choice)
        except Exception as e:
            st.session_state.last_exception = repr(e)
            st.error("Failed to load selected bank module / functions.")
            st.exception(e)
            return

        progress = st.progress(0)
        for i, uf in enumerate(uploaded_files, start=1):
            fname = uf.name
            try:
                raw_bytes = uf.getvalue()

                # decrypt if needed
                if is_pdf_encrypted(raw_bytes):
                    raw_bytes = decrypt_pdf_bytes(raw_bytes, pdf_password)

                # statement totals (optional)
                totals = _extract_statement_totals(totals_fn, raw_bytes)
                if totals:
                    st.session_state.statement_totals_by_file[fname] = totals

                # parse transactions
                txns_raw = _parse_transactions(parse_fn, mode, raw_bytes, fname)
                txns = normalize_transactions(txns_raw, default_bank=bank_choice, source_file=fname)

                # dedupe
                if bank_choice == "Affin Bank":
                    txns = dedupe_transactions_affin(txns)
                else:
                    txns = dedupe_transactions(txns)

                # OCR fallback only when needed
                if force_ocr or _count_real_transactions(txns) == 0:
                    try:
                        ocr_mod = importlib.import_module("ocr_fallback")
                        pages_text = ocr_mod.ocr_pdf_to_text_pages(raw_bytes, dpi=220)
                        ocr_raw = ocr_mod.parse_transactions_from_ocr_text_pages(pages_text, bank_choice, fname)
                        ocr_txns = normalize_transactions(ocr_raw, default_bank=bank_choice, source_file=fname)

                        if bank_choice == "Affin Bank":
                            txns = dedupe_transactions_affin(txns + ocr_txns)
                        else:
                            txns = dedupe_transactions(txns + ocr_txns)
                    except Exception as oe:
                        st.session_state.errors.append({"file": fname, "error": f"OCR failed: {oe}"})

                st.session_state.results.extend(txns)

            except Exception as e:
                st.session_state.errors.append({"file": fname, "error": str(e)})
                st.session_state.last_exception = repr(e)

            progress.progress(i / max(1, len(uploaded_files)))

        progress.empty()
        st.success("Processing completed.")

    # Show errors
    if st.session_state.errors:
        st.error("Some files failed to process:")
        st.json(st.session_state.errors)

    # Show results
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

    # Diagnostics (so you can actually see what's breaking)
    with st.expander("Diagnostics / Last Exception"):
        st.write("Last exception (if any):")
        st.code(st.session_state.last_exception or "None")


if __name__ == "__main__":
    main()
