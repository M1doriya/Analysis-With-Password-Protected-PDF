# public_bank.py - Standalone Public Bank Parser
import re

# ---------------------------------------------------------
# YEAR EXTRACTION
# ---------------------------------------------------------

def extract_year_from_text(text):
    """
    Extract year from Public Bank / Public Islamic Bank statement text.
    Safely handles:
    - Statement Date 31 Jul 2024
    - STATEMENT DATE : 30/09/24
    - Statement Date: DD/MM/YYYY
    - FOR THE PERIOD : DD/MM/YYYY
    """

    # -------------------------------------------------
    # Pattern 1: Statement Date 31 Jul 2024 (MOST COMMON)
    # -------------------------------------------------
    match = re.search(
        r'(?:Statement Date|Tarikh Penyata)\s*[:\s]+\d{1,2}\s+[A-Za-z]{3,}\s+(\d{4})',
        text,
        re.IGNORECASE
    )
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2100:
            return str(year)

    # -------------------------------------------------
    # Pattern 2: STATEMENT DATE : 30/09/24 or 30/09/2024
    # -------------------------------------------------
    match = re.search(
        r'(?:STATEMENT DATE|TARIKH PENYATA)\s*[:\s]+\d{1,2}/\d{1,2}/(\d{2,4})',
        text,
        re.IGNORECASE
    )
    if match:
        year_str = match.group(1)
        if len(year_str) == 4:
            year = int(year_str)
        else:
            year = 2000 + int(year_str)

        if 2000 <= year <= 2100:
            return str(year)

    # -------------------------------------------------
    # Pattern 3: Statement Date: DD/MM/YYYY
    # -------------------------------------------------
    match = re.search(
        r'Statement\s+(?:Date|Period)\s*[:\s]+\d{1,2}/\d{1,2}/(\d{4})',
        text,
        re.IGNORECASE
    )
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2100:
            return str(year)

    # -------------------------------------------------
    # Pattern 4: FOR THE PERIOD : DD/MM/YYYY
    # -------------------------------------------------
    match = re.search(
        r'FOR\s+THE\s+PERIOD\s*[:\s]+\d{1,2}/\d{1,2}/(\d{4})',
        text,
        re.IGNORECASE
    )
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2100:
            return str(year)

    # -------------------------------------------------
    # No safe year found
    # -------------------------------------------------
    return None



# ---------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------

# Matches date at start of line: "05/06 ..."
DATE_LINE = re.compile(r"^(?P<date>\d{2}/\d{2})\s+(?P<rest>.*)$")

# Matches amount + balance at end of line: "1,200.00 45,000.00"
AMOUNT_BAL = re.compile(r"(?P<amount>\d{1,3}(?:,\d{3})*\.\d{2})\s+(?P<balance>\d{1,3}(?:,\d{3})*\.\d{2})$")

# Matches "Balance B/F" lines
BAL_ONLY = re.compile(r"^(?P<date>\d{2}/\d{2})\s+(Balance.*)\s+(?P<balance>\d{1,3}(?:,\d{3})*\.\d{2})$", re.IGNORECASE)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TX_KEYWORDS = [
    "TSFR", "DUITNOW", "GIRO", "JOMPAY", "RMT", "DR-ECP",
    "HANDLING", "FEE", "DEP", "RTN", "PROFIT", "AUTOMATED",
    "CHARGES", "DEBIT", "CREDIT", "TRANSFER", "PAYMENT"
]

IGNORE_PREFIXES = [
    "CLEAR WATER", "/ROC", "PVCWS", "IMEPS", 
    "PUBLIC BANK", "PAGE", "TEL:", "MUKA SURAT", "TARIKH", 
    "DATE", "NO.", "URUS NIAGA", "STATEMENT", "ACCOUNT"
]


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def is_ignored(line):
    """Check if line should be ignored"""
    line_upper = line.upper()
    return any(line_upper.startswith(p) for p in IGNORE_PREFIXES)

def is_tx_start(line):
    """Check if line starts a new transaction"""
    return any(line.upper().startswith(k) for k in TX_KEYWORDS)


# ---------------------------------------------------------
# Main Parser
# ---------------------------------------------------------

def parse_transactions_pbb(pdf, source_filename=""):
    """
    Main parser for Public Bank statements.
    Automatically extracts year and parses all transactions.
    
    Args:
        pdf: pdfplumber PDF object
        source_filename: Name of the source file
    
    Returns:
        List of transaction dictionaries
    """
    all_transactions = []
    detected_year = None
    
    # Extract year from first few pages
    for page in pdf.pages[:3]:
        text = page.extract_text() or ""
        detected_year = extract_year_from_text(text)
        if detected_year:
            break
    
    # Fallback to current year
    if not detected_year:
        from datetime import datetime
        detected_year = str(datetime.now().year)
    
    # Process all pages
    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        
        tx = []
        current_date = None
        prev_balance = None
        desc_accum = ""
        waiting_for_amount = False
        
        lines = text.splitlines()
        
        for line in lines:
            line = line.strip()
            if not line or is_ignored(line):
                continue
            
            # Check for amounts FIRST
            amount_match = AMOUNT_BAL.search(line)
            has_amount = bool(amount_match)
            
            # Check for start of new transaction
            date_match = DATE_LINE.match(line)
            keyword_match = is_tx_start(line)
            is_new_start = date_match or keyword_match
            
            # Handle Balance B/F
            bal_match = BAL_ONLY.match(line)
            if bal_match:
                current_date = bal_match.group("date")
                prev_balance = float(bal_match.group("balance").replace(",", ""))
                desc_accum = ""
                waiting_for_amount = False
                continue
            
            # CASE A: Line has amounts
            if has_amount:
                amount = float(amount_match.group("amount").replace(",", ""))
                balance = float(amount_match.group("balance").replace(",", ""))
                
                # Determine description
                if is_new_start:
                    if date_match:
                        current_date = date_match.group("date")
                        line_desc = date_match.group("rest")
                    else:
                        line_desc = line.replace(amount_match.group(0), "").strip()
                    final_desc = line_desc
                else:
                    final_desc = desc_accum + " " + line.replace(amount_match.group(0), "").strip()
                
                # Determine debit vs credit
                debit = 0.0
                credit = 0.0
                
                if prev_balance is not None:
                    if balance < prev_balance:
                        debit = amount
                    elif balance > prev_balance:
                        credit = amount
                else:
                    # Fallback based on keywords
                    upper_desc = final_desc.upper()
                    if "CR" in upper_desc or "DEP" in upper_desc or "CREDIT" in upper_desc:
                        credit = amount
                    else:
                        debit = amount
                
                # Format date
                if current_date:
                    dd, mm = current_date.split("/")
                    iso_date = f"{detected_year}-{mm}-{dd}"
                else:
                    iso_date = f"{detected_year}-01-01"
                
                # Append transaction
                tx.append({
                    "date": iso_date,
                    "description": final_desc.strip(),
                    "debit": debit,
                    "credit": credit,
                    "balance": balance,
                    "page": page_num,
                    "source_file": source_filename,
                    "bank": "Public Bank"
                })
                
                # Reset state
                prev_balance = balance
                desc_accum = ""
                waiting_for_amount = False
            
            # CASE B: No amounts, but starts new transaction
            elif is_new_start:
                if date_match:
                    current_date = date_match.group("date")
                    desc_accum = date_match.group("rest")
                else:
                    desc_accum = line
                waiting_for_amount = True
            
            # CASE C: Continuation text
            elif waiting_for_amount:
                desc_accum += " " + line
        
        all_transactions.extend(tx)
    
    return all_transactions
