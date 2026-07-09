import os
import re
import json
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic_extract")

app = FastAPI(
    title="Dynamic Schema Structured Extraction API",
    description="Extracts structured data from raw text at runtime matching a caller-defined schema",
    version="1.0.0"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response schemas
class ExtractRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    text: str
    schema_dict: Dict[str, str] = Field(..., alias="schema")

# Date parser utility
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
    if not text:
        return None
    text = str(text).strip()
    
    # 1. ISO format YYYY-MM-DD
    m1 = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', text)
    if m1:
        return f"{m1.group(1)}-{m1.group(2)}-{m1.group(3)}"
        
    # 2. DD Month YYYY (e.g., 12 June 2026 or 12th June 2026)
    m2 = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]+)\s+(\d{4})\b', text)
    if m2:
        d, m_str, y = m2.group(1), m2.group(2).lower(), m2.group(3)
        if m_str in MONTHS_MAP:
            return f"{y}-{MONTHS_MAP[m_str]}-{int(d):02d}"
            
    # 3. Month DD, YYYY (e.g., June 12, 2026 or June 12th, 2026)
    m3 = re.search(r'\b([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b', text)
    if m3:
        m_str, d, y = m3.group(1).lower(), m3.group(2), m3.group(3)
        if m_str in MONTHS_MAP:
            return f"{y}-{MONTHS_MAP[m_str]}-{int(d):02d}"

    # 4. YYYY/MM/DD
    m4 = re.search(r'\b(\d{4})/(\d{1,2})/(\d{1,2})\b', text)
    if m4:
        return f"{m4.group(1)}-{int(m4.group(2)):02d}-{int(m4.group(3)):02d}"

    # 5. DD/MM/YYYY
    m5 = re.search(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', text)
    if m5:
        val1, val2, y = int(m5.group(1)), int(m5.group(2)), m5.group(3)
        if val1 > 12:
            return f"{y}-{val2:02d}-{val1:02d}"
        elif val2 > 12:
            return f"{y}-{val1:02d}-{val2:02d}"
        else:
            return f"{y}-{val2:02d}-{val1:02d}"
            
    return None

# Type coercion helpers
def coerce_integer(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        s = val.replace(",", "").strip()
        try:
            return int(s)
        except ValueError:
            try:
                return int(float(s))
            except ValueError:
                match = re.search(r'-?\d+', s)
                if match:
                    return int(match.group(0))
    return None

def coerce_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            match = re.search(r'-?\d+(?:\.\d+)?', s)
            if match:
                return float(match.group(0))
    return None

def coerce_boolean(val: Any) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "yes", "1", "t", "y", "on"):
            return True
        if s in ("false", "no", "0", "f", "n", "off"):
            return False
    return None

def coerce_array_string(val: Any) -> Optional[List[str]]:
    if val is None:
        return None
    if isinstance(val, list):
        return [str(x) for x in val if x is not None]
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                lst = json.loads(val_str)
                if isinstance(lst, list):
                    return [str(x) for x in lst if x is not None]
            except Exception:
                pass
        return [x.strip() for x in val_str.split(",") if x.strip()]
    return [str(val)]

def coerce_array_integer(val: Any) -> Optional[List[int]]:
    if val is None:
        return None
    if isinstance(val, list):
        res = []
        for x in val:
            coerced = coerce_integer(x)
            if coerced is not None:
                res.append(coerced)
        return res
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                lst = json.loads(val_str)
                if isinstance(lst, list):
                    res = []
                    for x in lst:
                        coerced = coerce_integer(x)
                        if coerced is not None:
                            res.append(coerced)
                    return res
            except Exception:
                pass
        res = []
        for x in val_str.split(","):
            coerced = coerce_integer(x)
            if coerced is not None:
                res.append(coerced)
        return res
    coerced = coerce_integer(val)
    return [coerced] if coerced is not None else None

def coerce_type(val: Any, expected_type: str) -> Any:
    expected_type = expected_type.strip().lower()
    if expected_type == "string":
        if val is None:
            return None
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        return str(val)
    elif expected_type == "integer":
        return coerce_integer(val)
    elif expected_type == "float":
        return coerce_float(val)
    elif expected_type == "boolean":
        return coerce_boolean(val)
    elif expected_type == "date":
        if val is None:
            return None
        return parse_date_to_iso(val)
    elif expected_type == "array[string]":
        return coerce_array_string(val)
    elif expected_type == "array[integer]":
        return coerce_array_integer(val)
    else:
        return val

def build_prompt(text: str, schema: Dict[str, str]) -> str:
    schema_desc = json.dumps(schema, indent=2)
    prompt = f"""You are a precise data extraction assistant.
You are given a raw TEXT and a SCHEMA defining the fields to extract and their types.
Your task is to extract information from the TEXT and format it according to the SCHEMA.

TEXT:
{text}

SCHEMA:
{schema_desc}

INSTRUCTIONS:
1. Extract the fields defined in the SCHEMA from the TEXT.
2. You must return a valid JSON object.
3. The JSON object must contain exactly the keys defined in the SCHEMA.
4. Do not include any additional keys or properties.
5. If a field cannot be extracted from the text, set its value to null.
6. Return only the JSON object. Do not include any explanations, markdown code blocks, or comments.
"""
    return prompt

@app.post("/dynamic-extract")
async def dynamic_extract(req: ExtractRequest):
    # Validate payload
    if not req.text or not req.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text is empty"
        )
    if not req.schema_dict:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Schema is empty"
        )

    # Read GEMINI_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API Key is not configured on the server. Please set the GEMINI_API_KEY environment variable."
        )

    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    prompt = build_prompt(req.text, req.schema_dict)

    request_payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json"
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(gemini_url, json=request_payload)
            
            if response.status_code != 200:
                logger.error(f"Gemini API error (HTTP {response.status_code}): {response.text}")
                err_detail = "Failed to communicate with Gemini API."
                try:
                    err_json = response.json()
                    if "error" in err_json and "message" in err_json["error"]:
                        err_detail = f"Gemini API Error: {err_json['error']['message']}"
                except Exception:
                    pass
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=err_detail
                )
                
            resp_data = response.json()
            candidates = resp_data.get("candidates", [])
            if not candidates:
                logger.warning("Gemini returned no candidates in response.")
                # Fallback: return all nulls
                return {field: None for field in req.schema_dict}
                
            first_candidate = candidates[0]
            parts = first_candidate.get("content", {}).get("parts", [])
            if not parts:
                logger.warning("Gemini candidate contains no parts.")
                return {field: None for field in req.schema_dict}
                
            raw_answer = parts[0].get("text", "")
            logger.info(f"Raw answer from Gemini: {raw_answer!r}")
            
            # Clean formatting just in case
            text_response = raw_answer.strip()
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.startswith("```"):
                text_response = text_response[3:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]
            text_response = text_response.strip()

            try:
                extracted_data = json.loads(text_response)
            except Exception as e:
                logger.error(f"Failed to parse LLM response as JSON: {text_response}")
                extracted_data = {}

            # Perform strict verification and type coercion
            output_data = {}
            for field, field_type in req.schema_dict.items():
                raw_val = extracted_data.get(field, None)
                output_data[field] = coerce_type(raw_val, field_type)

            return output_data

    except httpx.RequestError as exc:
        logger.error(f"Network error while connecting to Gemini: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Network error connecting to AI model: {str(exc)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Fallback to returning nulls for all fields
        return {field: None for field in req.schema_dict}

# Helper functions for structured extraction API
def word_to_number(text: str) -> int:
    text = text.lower().replace('-', ' ').replace(',', ' ')
    words = text.split()
    
    units = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }
    
    scales = {
        "hundred": 100,
        "thousand": 1000,
        "million": 1000000,
        "billion": 1000000000,
        "lakh": 100000,
        "crore": 10000000
    }
    
    current = 0
    result = 0
    
    for word in words:
        if word in units:
            current += units[word]
        elif word in scales:
            scale = scales[word]
            if scale >= 100:
                if current == 0:
                    current = 1
                current *= scale
                if scale >= 1000:
                    result += current
                    current = 0
            else:
                current *= scale
        elif word == "and":
            continue
        else:
            try:
                current += int(word)
            except ValueError:
                pass
                
    return result + current

def parse_amount_to_int(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return int(val)
        
    s = str(val).strip().lower()
    
    if s.endswith('k'):
        s_num = s[:-1].strip()
        try:
            return int(float(s_num) * 1000)
        except ValueError:
            pass
    if s.endswith('m'):
        s_num = s[:-1].strip()
        try:
            return int(float(s_num) * 1000000)
        except ValueError:
            pass
            
    # Try cleaning standard formatting like $12,480.00
    cleaned = re.sub(r'[^\d.]', '', s.replace(',', ''))
    try:
        if '.' in cleaned:
            return int(float(cleaned))
        return int(cleaned)
    except ValueError:
        pass
        
    try:
        word_val = word_to_number(s)
        if word_val > 0:
            return word_val
    except Exception:
        pass
        
    # Extract first sequence of digits
    match = re.search(r'\d+', s.replace(',', ''))
    if match:
        return int(match.group(0))
        
    return 0

def normalize_currency(val: Any) -> str:
    if not val:
        return "USD"
    s = str(val).strip().lower()
    
    mapping = {
        "usd": "USD", "dollar": "USD", "$": "USD",
        "eur": "EUR", "euro": "EUR", "€": "EUR",
        "gbp": "GBP", "pound": "GBP", "sterling": "GBP", "£": "GBP",
        "inr": "INR", "rupee": "INR", "₹": "INR", "rs": "INR",
        "jpy": "JPY", "yen": "JPY", "¥": "JPY"
    }
    
    for k, v in mapping.items():
        if k in s:
            return v
            
    return val.strip().upper() if isinstance(val, str) else "USD"

def normalize_date_to_iso(val: Any) -> str:
    if not val:
        return ""
    
    s = str(val).strip()
    m = re.match(r'^\d{4}-\d{2}-\d{2}$', s)
    if m:
        return s
        
    parsed = parse_date_to_iso(s)
    if parsed:
        return parsed
        
    return s

def parse_due_in_days(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return int(val)
        
    s = str(val).strip().lower()
    
    try:
        return int(s)
    except ValueError:
        pass
        
    m_weeks = re.search(r'(\d+)\s*weeks?', s)
    if m_weeks:
        return int(m_weeks.group(1)) * 7
    if "two weeks" in s:
        return 14
    if "a week" in s or "one week" in s:
        return 7
        
    m_days = re.search(r'(\d+)\s*days?', s)
    if m_days:
        return int(m_days.group(1))
        
    m_net = re.search(r'net\s*(\d+)', s)
    if m_net:
        return int(m_net.group(1))
        
    try:
        word_val = word_to_number(s)
        if word_val > 0:
            return word_val
    except Exception:
        pass
        
    m_num = re.search(r'\d+', s)
    if m_num:
        return int(m_num.group(0))
        
    return 0

def parse_is_paid(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
        
    s = str(val).strip().lower()
    if "paid" in s or "payment received" in s or "settled" in s or "yes" in s or "true" in s or "1" in s:
        if "awaiting" in s or "not paid" in s or "unpaid" in s:
            return False
        return True
    return False

def parse_priority(val: Any) -> str:
    if not val:
        return "normal"
    s = str(val).strip().lower()
    if s in ["low", "normal", "high", "urgent"]:
        return s
    if "urg" in s or "crit" in s or "immediate" in s:
        return "urgent"
    if "high" in s or "important" in s:
        return "high"
    if "low" in s or "minor" in s:
        return "low"
    return "normal"

def parse_contact_email(val: Any) -> str:
    if not val:
        return ""
    return str(val).strip().lower()

def parse_line_items(val: Any) -> list:
    if not isinstance(val, list):
        return []
    cleaned_items = []
    for item in val:
        if not isinstance(item, dict):
            continue
        cleaned_item = {
            "sku": str(item.get("sku", "")).strip(),
            "quantity": coerce_integer(item.get("quantity", 0)) or 0,
            "unit_price": coerce_integer(item.get("unit_price", 0)) or 0
        }
        cleaned_items.append(cleaned_item)
    return cleaned_items

def clean_for_gemini_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
        
    allowed_keys = {
        "type", "properties", "items", "required", "description", "enum", "format", "nullable"
    }
    
    cleaned = {}
    for k, v in schema.items():
        if k not in allowed_keys:
            continue
            
        if k == "type" and isinstance(v, str):
            cleaned[k] = v.upper()
        elif k == "properties" and isinstance(v, dict):
            cleaned[k] = {prop_name: clean_for_gemini_schema(prop_val) for prop_name, prop_val in v.items()}
        elif k == "items" and isinstance(v, dict):
            cleaned[k] = clean_for_gemini_schema(v)
        elif k in ("required", "description", "enum", "format", "nullable"):
            cleaned[k] = v
            
    return cleaned

def post_process_extracted_data(data: dict, schema: dict) -> dict:
    properties = schema.get("properties", {})
    output = {}
    
    for field_name, field_schema in properties.items():
        val = data.get(field_name, None)
        
        if field_name == "vendor":
            output[field_name] = str(val).strip() if val is not None else ""
        elif field_name == "currency":
            output[field_name] = normalize_currency(val)
        elif field_name == "total_amount":
            output[field_name] = parse_amount_to_int(val)
        elif field_name == "invoice_date":
            parsed_date = normalize_date_to_iso(val)
            output[field_name] = parsed_date if parsed_date else ""
        elif field_name == "due_in_days":
            output[field_name] = parse_due_in_days(val)
        elif field_name == "is_paid":
            output[field_name] = parse_is_paid(val)
        elif field_name == "priority":
            output[field_name] = parse_priority(val)
        elif field_name == "contact_email":
            output[field_name] = parse_contact_email(val)
        elif field_name == "line_items":
            output[field_name] = parse_line_items(val)
        elif field_name == "item_count":
            items = output.get("line_items") or parse_line_items(data.get("line_items"))
            output[field_name] = len(items)
        else:
            expected_type = field_schema.get("type", "string").lower()
            if expected_type == "integer":
                output[field_name] = coerce_integer(val)
            elif expected_type == "float" or expected_type == "number":
                output[field_name] = coerce_float(val)
            elif expected_type == "boolean":
                output[field_name] = coerce_boolean(val)
            elif expected_type == "array":
                if isinstance(val, list):
                    output[field_name] = val
                else:
                    output[field_name] = []
            else:
                output[field_name] = str(val) if val is not None else None
                
    if "item_count" in properties and "line_items" in output:
        output["item_count"] = len(output["line_items"])
        
    return output

class InvoiceExtractionRequest(BaseModel):
    document_id: Optional[str] = None
    text: str
    schema_dict: Dict[str, Any] = Field(..., alias="schema")

@app.post("/extract")
@app.post("/")
async def extract_invoice(req: InvoiceExtractionRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text is empty"
        )
    if not req.schema_dict:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Schema is empty"
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API Key is not configured on the server. Please set the GEMINI_API_KEY environment variable."
        )

    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    gemini_schema = clean_for_gemini_schema(req.schema_dict)

    prompt = f"""You are a precise invoice extraction assistant.
You are given a raw invoice text and a schema.
Your task is to extract the fields matching the schema exactly from the text.

Here are the details of how to map the fields:
- vendor: the biller's proper name, exactly as written in the text.
- currency: the ISO 4217 code (USD, EUR, GBP, INR, JPY) — map symbols like $, €, £, ₹ or words to their codes.
- total_amount: integer in the main unit (no separators or symbols, e.g., 12480 instead of 12,480). Translate suffixes like K or M (e.g. 12K -> 12000). Translate spelled-out numbers like "twelve thousand four hundred eighty" to 12480.
- invoice_date: normalize to YYYY-MM-DD.
- due_in_days: integer (e.g. "Net 30" -> 30, "payable within 45 days" -> 45, "due in two weeks" -> 14).
- is_paid: boolean inferred from wording (e.g., "paid in full" -> true, "awaiting payment" -> false).
- priority: one of low, normal, high, urgent.
- contact_email: lowercased email address.
- line_items: array of items containing sku, quantity (integer), unit_price (integer) in the order they appear.
- item_count: number of line items (integer).

Invoice Text:
{req.text}
"""

    request_payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "responseSchema": gemini_schema
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(gemini_url, json=request_payload)
            
            if response.status_code != 200:
                logger.error(f"Gemini API error (HTTP {response.status_code}): {response.text}")
                err_detail = "Failed to communicate with Gemini API."
                try:
                    err_json = response.json()
                    if "error" in err_json and "message" in err_json["error"]:
                        err_detail = f"Gemini API Error: {err_json['error']['message']}"
                except Exception:
                    pass
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=err_detail
                )
                
            resp_data = response.json()
            candidates = resp_data.get("candidates", [])
            if not candidates:
                logger.warning("Gemini returned no candidates in response.")
                return post_process_extracted_data({}, req.schema_dict)
                
            first_candidate = candidates[0]
            parts = first_candidate.get("content", {}).get("parts", [])
            if not parts:
                logger.warning("Gemini candidate contains no parts.")
                return post_process_extracted_data({}, req.schema_dict)
                
            raw_answer = parts[0].get("text", "")
            logger.info(f"Raw answer from Gemini: {raw_answer!r}")
            
            text_response = raw_answer.strip()
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.startswith("```"):
                text_response = text_response[3:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]
            text_response = text_response.strip()

            try:
                extracted_data = json.loads(text_response)
            except Exception as e:
                logger.error(f"Failed to parse LLM response as JSON: {text_response}")
                extracted_data = {}

            output_data = post_process_extracted_data(extracted_data, req.schema_dict)
            return output_data

    except httpx.RequestError as exc:
        logger.error(f"Network error while connecting to Gemini: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Network error connecting to AI model: {str(exc)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return post_process_extracted_data({}, req.schema_dict)

@app.get("/")
def root():
    return {"status": "ok", "service": "dynamic-schema-extraction"}

