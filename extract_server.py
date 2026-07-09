"""
Invoice Field Extraction Service
POST /extract — extracts vendor, amount, currency, date, invoice_no, tax from free-form invoice text.
Uses robust regex-based extraction; no heavy ML model download required.
"""
import re
import dateutil.parser
from typing import Optional, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="Invoice Extraction Service",
    description="Extracts structured invoice fields from free-form text",
    version="1.0.0",
)

# Configure CORS Middleware
# To support cross-origin requests from any source, including Cloudflare Workers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ExtractionRequest(BaseModel):
    invoice_text: Optional[str] = None
    text: Optional[str] = None  # Fallback field name

class ExtractionResponse(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = None
    vendor: Optional[str] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

MONTHS_MAP = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12"
}

def parse_date_to_iso(text: str) -> Optional[str]:
    # 1. First, search for a date string after "date" keyword
    date_keyword_match = re.search(
        r'\b(?:date|dated|due|issued|issue)[:\s]+([a-zA-Z0-9\s,./-]{6,30})',
        text,
        re.IGNORECASE
    )
    if date_keyword_match:
        cand = date_keyword_match.group(1).strip()
        # Clean trailing noise/labels
        cand = re.split(r'\n|\r|total|amount|invoice|inv|vendor|seller|customer|to:|\b[a-zA-Z]{4,}\b', cand, flags=re.IGNORECASE)[0].strip()
        cand = cand.strip(' \t\n\r"\'.,:-/')
        if len(cand) >= 6:
            try:
                dt = dateutil.parser.parse(cand, fuzzy=True)
                if 2020 <= dt.year <= 2035:
                    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
            except Exception:
                pass

    # 2. Try explicit regex patterns for standard formats
    # Pattern 2.1: YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
    m1 = re.search(r'\b(\d{4})[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\b', text)
    if m1:
        return f"{int(m1.group(1)):04d}-{int(m1.group(2)):02d}-{int(m1.group(3)):02d}"

    # Pattern 2.2: DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY
    m2 = re.search(r'\b(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.](\d{4})\b', text)
    if m2:
        d, m, y = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        return f"{y:04d}-{m:02d}-{d:02d}"

    # Pattern 2.3: DD-MM-YY or DD/MM/YY (2-digit year)
    m3 = re.search(r'\b(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.](\d{2})\b', text)
    if m3:
        d, m, y_short = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
        y = 2000 + y_short if y_short < 50 else 1900 + y_short
        return f"{y:04d}-{m:02d}-{d:02d}"

    # Pattern 2.4: Written date formats: Month DD, YYYY or DD Month YYYY
    m4 = re.search(
        r'\b([a-zA-Z]{3,})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
        text,
        re.IGNORECASE
    )
    if m4:
        m_str, d, y = m4.group(1).lower()[:3], int(m4.group(2)), int(m4.group(3))
        if m_str in MONTHS_MAP:
            return f"{y:04d}-{int(MONTHS_MAP[m_str]):02d}-{d:02d}"

    m5 = re.search(
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]{3,})\s+(\d{4})\b',
        text,
        re.IGNORECASE
    )
    if m5:
        d, m_str, y = int(m5.group(1)), m5.group(2).lower()[:3], int(m5.group(3))
        if m_str in MONTHS_MAP:
            return f"{y:04d}-{int(MONTHS_MAP[m_str]):02d}-{d:02d}"

    m6 = re.search(
        r'\b(\d{1,2})(?:st|nd|rd|th)?[-/]([a-zA-Z]{3,})[-/](\d{2,4})\b',
        text,
        re.IGNORECASE
    )
    if m6:
        d, m_str, y_str = int(m6.group(1)), m6.group(2).lower()[:3], m6.group(3)
        y = int(y_str)
        if len(y_str) == 2:
            y = 2000 + y
        if m_str in MONTHS_MAP:
            return f"{y:04d}-{int(MONTHS_MAP[m_str]):02d}-{d:02d}"

    # 3. Fallback: scan for any string that dateutil can parse
    words = re.findall(r'\b[a-zA-Z0-9\s,./-]{6,25}\b', text)
    for w in words:
        w_clean = w.strip(' \t\n\r"\'.,:-/')
        if len(w_clean) >= 8:
            try:
                dt = dateutil.parser.parse(w_clean, fuzzy=True)
                if 2020 <= dt.year <= 2035:
                    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
            except Exception:
                pass

    return None

def parse_currency(text: str) -> Optional[str]:
    m = re.search(r'\b(INR|USD|EUR|GBP|CAD|AUD|SGD|JPY)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    if '₹' in text or 'Rs.' in text or 'Rs' in text or 'rupees' in text.lower():
        return 'INR'
    if '$' in text:
        return 'USD'
    if '€' in text:
        return 'EUR'
    if '£' in text:
        return 'GBP'
        
    return None

def extract_float(s: str) -> Optional[float]:
    cleaned = re.sub(r'[^\d.]', '', s.replace(',', ''))
    try:
        return float(cleaned)
    except ValueError:
        return None

def parse_amount_and_tax(text: str) -> Tuple[Optional[float], Optional[float]]:
    subtotal = None
    tax = None
    total = None
    
    # Extract all candidate prices first
    prices = re.findall(r'\b(?:[A-Za-z.$€£₹]+\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d{2}))\b', text)
    floats = []
    for p in prices:
        val = extract_float(p)
        if val is not None and val not in (2025.0, 2026.0, 2027.0):
            floats.append(val)
            
    # Search for Subtotal keyword
    subtotal_match = re.search(
        r'\b(?:sub[- ]?total|net[- ]?amount|before[- ]?tax)[:\s]*([A-Za-z.$€£₹]*\s*[\d,]+(?:\.\d+)?)\b',
        text,
        re.IGNORECASE
    )
    if subtotal_match:
        subtotal = extract_float(subtotal_match.group(1))
        
    # Search for Tax keyword
    tax_match = re.search(
        r'\b(?:gst|vat|sales[- ]?tax|service[- ]?tax|tax|cgst|sgst|igst)(?:\s*\(\d+%\))?[:\s]*([A-Za-z.$€£₹]*\s*[\d,]+(?:\.\d+)?)\b',
        text,
        re.IGNORECASE
    )
    if tax_match:
        tax = extract_float(tax_match.group(1))
        
    # Search for Total keyword (including just "Amount:")
    total_match = re.search(
        r'\b(?:total|grand[- ]?total|amount[- ]?due|balance[- ]?due|payable|amount)[:\s]*([A-Za-z.$€£₹]*\s*[\d,]+(?:\.\d+)?)\b',
        text,
        re.IGNORECASE
    )
    if total_match:
        total = extract_float(total_match.group(1))
        
    # Mathematical reconstruction/validation
    if subtotal is None and total is not None and tax is not None:
        subtotal = round(total - tax, 2)
        
    if tax is None and total is not None and subtotal is not None:
        tax = round(total - subtotal, 2)
        
    if subtotal is None or tax is None:
        if len(floats) >= 3:
            sorted_floats = sorted(list(set(floats)))
            if len(sorted_floats) >= 3:
                for i in range(len(sorted_floats)):
                    for j in range(i + 1, len(sorted_floats)):
                        for k in range(j + 1, len(sorted_floats)):
                            v_small = sorted_floats[i]
                            v_med = sorted_floats[j]
                            v_large = sorted_floats[k]
                            if abs(v_large - (v_med + v_small)) < 0.1:
                                if subtotal is None:
                                    subtotal = v_med
                                if tax is None:
                                    tax = v_small
                                break
                                
    # Final fallbacks
    if subtotal is None:
        if total is not None:
            subtotal = total
        elif floats:
            subtotal = floats[0]
            
    if tax is None:
        tax = 0.0
        
    return subtotal, tax

def parse_vendor(text: str) -> Optional[str]:
    # 1. vendor/seller/supplier lines
    m = re.search(
        r'\b(?:vendor|seller|supplier|company|from|issued\s+by)[:\s]+([^\n]+)',
        text,
        re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip().strip(' \t\n\r"\'.,:-')
        if candidate and not any(k in candidate.lower() for k in ["bill to", "invoice", "date", "total", "amount"]):
            return candidate

    # 2. Look for lines ending with Pvt Ltd, Ltd, Inc, Corp, LLC, GmbH, Co.
    for line in text.split('\n'):
        line_clean = line.strip().strip(' \t\n\r"\'.,:-')
        if re.search(r'\b(?:Pvt\s+Ltd|Ltd|Inc|Corp|LLC|GmbH|Co|Corporation|Pvt\.?\s*Ltd\.?)\b', line_clean, re.IGNORECASE):
            cleaned_line = re.sub(r'^(?:vendor|seller|supplier|company|from|issued\s+by)[:\s]+', '', line_clean, flags=re.IGNORECASE)
            if not any(k in cleaned_line.lower() for k in ["bill to", "procurement", "buyer"]):
                return cleaned_line.strip()
                
    return None

def parse_invoice_no(text: str) -> Optional[str]:
    m1 = re.search(r'\b(?:invoice\s+(?:no\.?|number|#|code)|inv\s+(?:no\.?|#))[:\s]*([a-zA-Z0-9\-_]+)', text, re.IGNORECASE)
    if m1:
        return m1.group(1).strip()
        
    m2 = re.search(r'\b(INV-[a-zA-Z0-9\-_]+)\b', text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
        
    return None

def extract_fields(invoice_text: str) -> dict:
    subtotal, tax = parse_amount_and_tax(invoice_text)
    return {
        "invoice_no": parse_invoice_no(invoice_text),
        "date": parse_date_to_iso(invoice_text),
        "vendor": parse_vendor(invoice_text),
        "amount": subtotal,
        "tax": tax,
        "currency": parse_currency(invoice_text)
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "invoice-extractor"}

@app.post("/extract", response_model=ExtractionResponse, tags=["extraction"])
def extract(req: ExtractionRequest):
    """Extract structured invoice fields from free-form text."""
    invoice_text = req.invoice_text or req.text
    if not invoice_text or not invoice_text.strip():
        return JSONResponse(
            status_code=422,
            content={"detail": "Input text must be non-empty"},
        )
    try:
        result = extract_fields(invoice_text)
        return ExtractionResponse(**result)
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": f"Extraction failed: {str(exc)}"},
        )
