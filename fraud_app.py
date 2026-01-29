import json
from io import BytesIO
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from fraud_parser import parse_top_parties_and_high_value, parse_inter_transactions


st.set_page_config(page_title="Fraud Analyzer (JSON Upload)", layout="wide")
st.title("🕵️ Fraud Analyzer (JSON Upload)")
st.write(
    "Upload either **transactions.json** (a list of transactions) or **full_report.json** "
    "(an object containing a `transactions` field)."
)


def extract_transactions(payload: Any) -> Tuple[List[Dict], str]:
    """Return (transactions, detected_format)."""
    if isinstance(payload, list):
        return payload, "transactions_list"

    if isinstance(payload, dict):
        if isinstance(payload.get("transactions"), list):
            return payload["transactions"], "full_report"

        if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("transactions"), list):
            return payload["data"]["transactions"], "data.transactions"

    return [], "unknown"


uploaded = st.file_uploader("Upload JSON file", type=["json"], accept_multiple_files=False)

if uploaded is None:
    st.info("Upload a JSON file to begin.")
    st.stop()

try:
    payload = json.load(uploaded)
except Exception as e:
    st.error(f"❌ Could not read JSON: {e}")
    st.stop()

transactions, fmt = extract_transactions(payload)

if not transactions:
    st.error(
        "❌ I couldn't find transactions in this JSON. "
        "Expected either a **list** of transactions, or an **object** with a `transactions` list."
    )
    st.stop()

st.success(f"✅ Loaded {len(transactions)} transactions (format: {fmt})")

df = pd.DataFrame(transactions)
preferred_cols = ["date", "description", "debit", "credit", "balance", "page", "bank", "source_file"]
display_cols = [c for c in preferred_cols if c in df.columns] or list(df.columns)

st.subheader("📄 Transactions (Preview)")
st.dataframe(df[display_cols], use_container_width=True)

# ---------------------------------------------------
# CONTROLS
# ---------------------------------------------------
st.sidebar.header("⚙️ Fraud Analyzer Settings")

top_n = st.sidebar.number_input("Top N counterparties", min_value=1, max_value=50, value=5, step=1)
high_value_threshold = st.sidebar.number_input(
    "High-value credit threshold", min_value=0.0, value=100_000.0, step=10_000.0
)
threshold_mode = st.sidebar.selectbox("Threshold mode", ["gte", "lte"], index=0)

st.sidebar.markdown("---")
match_mode = st.sidebar.selectbox("Inter-trace match mode", ["any", "all", "min"], index=0)
min_token_matches = st.sidebar.number_input(
    "Min token matches (if match mode = min)", min_value=1, max_value=10, value=2, step=1
)

# ---------------------------------------------------
# FRAUD ANALYSIS
# ---------------------------------------------------
st.markdown("---")
st.subheader("🔎 Fraud Analysis")

st.markdown("### 1️⃣ Top Parties & High-Value Credits")
fraud_summary = parse_top_parties_and_high_value(
    transactions,
    top_n=int(top_n),
    high_value_threshold=float(high_value_threshold),
    threshold_mode=threshold_mode,
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🔝 Top Credit Parties")
    st.dataframe(pd.DataFrame(fraud_summary["top_credit_parties"]), use_container_width=True)
with col2:
    st.markdown("#### 🔻 Top Debit Parties")
    st.dataframe(pd.DataFrame(fraud_summary["top_debit_parties"]), use_container_width=True)

st.markdown("#### 💰 High-Value Credit Transactions")
if fraud_summary["high_value_credits"]:
    st.dataframe(pd.DataFrame(fraud_summary["high_value_credits"]), use_container_width=True)
else:
    st.info("No high-value credit transactions detected.")

st.markdown("---")
st.markdown("### 2️⃣ Inter-Transaction Trace")
company_name = st.text_input("🏢 Company name for tracing", placeholder="e.g. MAZA SDN BHD")

trace_result = None
if company_name.strip():
    trace_result = parse_inter_transactions(
        transactions,
        company_name,
        match_mode=match_mode,
        min_token_matches=int(min_token_matches),
    )

    st.markdown("#### Summary")
    st.json({
        "company": trace_result["company_name"],
        "transaction_count": trace_result["transaction_count"],
        "total_credit": trace_result["total_credit"],
        "total_debit": trace_result["total_debit"],
        "net_flow": trace_result["net_flow"],
        "company_tokens": trace_result["company_tokens"],
    })

    st.markdown("#### Matched Transactions")
    st.dataframe(pd.DataFrame(trace_result["transactions"]), use_container_width=True)
else:
    st.info("Enter a company name to run inter-transaction tracing.")

# ---------------------------------------------------
# DOWNLOADS
# ---------------------------------------------------
st.markdown("---")
st.subheader("⬇️ Download Options")

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        "📄 Download Fraud Summary (JSON)",
        data=json.dumps(fraud_summary, indent=2),
        file_name="fraud_summary.json",
        mime="application/json",
    )

with col2:
    if trace_result is not None:
        st.download_button(
            "🧾 Download Inter-Trace (JSON)",
            data=json.dumps(trace_result, indent=2),
            file_name=f"inter_trace_{company_name.strip().replace(' ', '_')}.json",
            mime="application/json",
        )
    else:
        st.caption("Inter-trace will appear after you enter a company name.")

with col3:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df[display_cols].to_excel(writer, sheet_name="Transactions", index=False)
        pd.DataFrame(fraud_summary["top_credit_parties"]).to_excel(writer, sheet_name="Top Credit Parties", index=False)
        pd.DataFrame(fraud_summary["top_debit_parties"]).to_excel(writer, sheet_name="Top Debit Parties", index=False)
        pd.DataFrame(fraud_summary["high_value_credits"]).to_excel(writer, sheet_name="High Value Credits", index=False)
        if trace_result is not None:
            pd.DataFrame(trace_result["transactions"]).to_excel(writer, sheet_name="Inter Trace", index=False)

    st.download_button(
        "📊 Download Fraud Report (XLSX)",
        data=output.getvalue(),
        file_name="fraud_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
