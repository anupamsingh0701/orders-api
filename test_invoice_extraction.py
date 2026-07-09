import json
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import os

# Set dummy key for environment verification
os.environ["GEMINI_API_KEY"] = "mock-api-key-for-testing"

from dynamic_extract import (
    app,
    word_to_number,
    parse_amount_to_int,
    normalize_currency,
    normalize_date_to_iso,
    parse_due_in_days,
    parse_is_paid,
    parse_priority,
    post_process_extracted_data,
    clean_for_gemini_schema
)

class TestInvoiceExtraction(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_word_to_number(self):
        self.assertEqual(word_to_number("twelve thousand four hundred eighty"), 12480)
        self.assertEqual(word_to_number("two million five hundred thousand"), 2500000)
        self.assertEqual(word_to_number("one hundred five"), 105)
        self.assertEqual(word_to_number("zero"), 0)

    def test_parse_amount_to_int(self):
        self.assertEqual(parse_amount_to_int("12K"), 12000)
        self.assertEqual(parse_amount_to_int("12.5k"), 12500)
        self.assertEqual(parse_amount_to_int("12,480"), 12480)
        self.assertEqual(parse_amount_to_int("1,24,800"), 124800)
        self.assertEqual(parse_amount_to_int("$ 12,480.00"), 12480)
        self.assertEqual(parse_amount_to_int("twelve thousand four hundred eighty"), 12480)
        self.assertEqual(parse_amount_to_int(500), 500)
        self.assertEqual(parse_amount_to_int(None), 0)

    def test_normalize_currency(self):
        self.assertEqual(normalize_currency("euros"), "EUR")
        self.assertEqual(normalize_currency("euro"), "EUR")
        self.assertEqual(normalize_currency("₹"), "INR")
        self.assertEqual(normalize_currency("rupees"), "INR")
        self.assertEqual(normalize_currency("pounds sterling"), "GBP")
        self.assertEqual(normalize_currency("GBP"), "GBP")
        self.assertEqual(normalize_currency("$"), "USD")
        self.assertEqual(normalize_currency("yen"), "JPY")
        self.assertEqual(normalize_currency("unknown"), "UNKNOWN")

    def test_normalize_date_to_iso(self):
        self.assertEqual(normalize_date_to_iso("2026-06-12"), "2026-06-12")
        self.assertEqual(normalize_date_to_iso("12 June 2026"), "2026-06-12")
        self.assertEqual(normalize_date_to_iso("June 12th, 2026"), "2026-06-12")
        self.assertEqual(normalize_date_to_iso("2026/06/12"), "2026-06-12")
        self.assertEqual(normalize_date_to_iso("12/06/2026"), "2026-06-12")

    def test_parse_due_in_days(self):
        self.assertEqual(parse_due_in_days("Net 30"), 30)
        self.assertEqual(parse_due_in_days("payable within 45 days"), 45)
        self.assertEqual(parse_due_in_days("due in two weeks"), 14)
        self.assertEqual(parse_due_in_days("due in a week"), 7)
        self.assertEqual(parse_due_in_days(15), 15)
        self.assertEqual(parse_due_in_days(None), 0)

    def test_parse_is_paid(self):
        self.assertTrue(parse_is_paid("paid in full"))
        self.assertTrue(parse_is_paid("payment received"))
        self.assertTrue(parse_is_paid("settled"))
        self.assertTrue(parse_is_paid(True))
        self.assertFalse(parse_is_paid("awaiting payment"))
        self.assertFalse(parse_is_paid("unpaid"))
        self.assertFalse(parse_is_paid(False))

    def test_parse_priority(self):
        self.assertEqual(parse_priority("low"), "low")
        self.assertEqual(parse_priority("urgent"), "urgent")
        self.assertEqual(parse_priority("critical"), "urgent")
        self.assertEqual(parse_priority("immediate"), "urgent")
        self.assertEqual(parse_priority("important"), "high")
        self.assertEqual(parse_priority("unknown"), "normal")

    def test_clean_for_gemini_schema(self):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "vendor": {"type": "string", "description": "vendor name"},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"}
                        }
                    }
                }
            },
            "additionalProperties": False
        }
        cleaned = clean_for_gemini_schema(schema)
        self.assertEqual(cleaned["type"], "OBJECT")
        self.assertNotIn("$schema", cleaned)
        self.assertNotIn("additionalProperties", cleaned)
        self.assertEqual(cleaned["properties"]["vendor"]["type"], "STRING")
        self.assertEqual(cleaned["properties"]["line_items"]["type"], "ARRAY")
        self.assertEqual(cleaned["properties"]["line_items"]["items"]["type"], "OBJECT")

    @patch("httpx.AsyncClient.post")
    def test_extract_endpoint(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "vendor": "Acme Industrial Supply",
                                    "currency": "euros",
                                    "total_amount": "twelve thousand four hundred eighty",
                                    "invoice_date": "June 12th, 2026",
                                    "due_in_days": "due in two weeks",
                                    "is_paid": "paid in full",
                                    "priority": "critical",
                                    "contact_email": "AP@ACME.COM",
                                    "line_items": [
                                        {"sku": "WIDGET-204", "quantity": "12", "unit_price": "40"},
                                        {"sku": "BOLT-118", "quantity": "200", "unit_price": "5"}
                                    ]
                                })
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        schema = {
            "type": "object",
            "properties": {
                "vendor": {"type": "string"},
                "currency": {"type": "string"},
                "total_amount": {"type": "integer"},
                "invoice_date": {"type": "string"},
                "due_in_days": {"type": "integer"},
                "is_paid": {"type": "boolean"},
                "priority": {"type": "string"},
                "contact_email": {"type": "string"},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "integer"}
                        }
                    }
                },
                "item_count": {"type": "integer"}
            }
        }

        payload = {
            "document_id": "doc0",
            "text": "Invoice text here...",
            "schema": schema
        }

        response = self.client.post("/extract", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["vendor"], "Acme Industrial Supply")
        self.assertEqual(data["currency"], "EUR")
        self.assertEqual(data["total_amount"], 12480)
        self.assertEqual(data["invoice_date"], "2026-06-12")
        self.assertEqual(data["due_in_days"], 14)
        self.assertTrue(data["is_paid"])
        self.assertEqual(data["priority"], "urgent")
        self.assertEqual(data["contact_email"], "ap@acme.com")
        self.assertEqual(len(data["line_items"]), 2)
        self.assertEqual(data["line_items"][0]["sku"], "WIDGET-204")
        self.assertEqual(data["line_items"][0]["quantity"], 12)
        self.assertEqual(data["line_items"][0]["unit_price"], 40)
        self.assertEqual(data["item_count"], 2)

if __name__ == "__main__":
    unittest.main()
