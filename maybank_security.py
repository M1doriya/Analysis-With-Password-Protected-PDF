import re

# =========================
# NORMALIZATION
# =========================
def normalize(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\b(SDN|BHD|BERHAD|LIMITED|LTD)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()

# =========================
# INTER-TRANSACTION CHECK
# =========================
def is_inter_transaction(company_name: str, description: str) -> bool:
    """
    Returns True if transaction description appears
    to match the company name (inter-transaction).
    """
    if not company_name or not description:
        return False

    comp = normalize(company_name)
    desc = normalize(description)

    tokens = comp.split()
    matches = sum(1 for t in tokens if t in desc)

    return matches >= 2   # safe threshold

# =========================
# APPLY SECURITY CHECK
# =========================
def apply_maybank_security(transactions: list, company_name: str, enabled: bool = True):
    """
    Adds 'fraud_flag' to each transaction dict.
    Does NOT modify extraction logic.
    """

    if not enabled or not company_name:
        for t in transactions:
            t["fraud_flag"] = False
        return transactions

    for t in transactions:
        desc = t.get("description", "")
        t["fraud_flag"] = is_inter_transaction(company_name, desc)

    return transactions
