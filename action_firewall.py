import os
import re
import html
import urllib.parse
from html.parser import HTMLParser
from typing import Any, Dict, Optional

ASSIGNED_TENANT = os.environ.get("ASSIGNED_TENANT", "tenant-2dtengp")
ASSIGNED_EMAIL_DOMAIN = os.environ.get("ASSIGNED_EMAIL_DOMAIN", "notify-n3gfzba.example")

ALLOWED_TOOLS = {"search", "lookup_record", "send_email", "render_html"}


class SafeHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.is_unsafe = False
        self.unsafe_reason = ""

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower in ("script", "iframe"):
            self.is_unsafe = True
            self.unsafe_reason = f"Forbidden tag <{tag_lower}>"
            return

        for attr_name, attr_value in attrs:
            attr_name_lower = attr_name.lower().strip()
            # Inline event handlers
            if attr_name_lower.startswith("on"):
                self.is_unsafe = True
                self.unsafe_reason = f"Inline event handler: {attr_name}"
                return

            # JavaScript URLs in attributes
            if attr_value:
                val_str = str(attr_value)
                val_unescaped = html.unescape(val_str)
                val_unquoted = urllib.parse.unquote(val_unescaped)
                val_clean = re.sub(r"[\s\x00-\x1f\x7f-\x9f]", "", val_unquoted.lower())
                if val_clean.startswith("javascript:") or "javascript:" in val_clean:
                    self.is_unsafe = True
                    self.unsafe_reason = f"JavaScript URL in attribute {attr_name}"
                    return

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)


def check_html_safety(raw_html: str) -> bool:
    """
    Checks if HTML contains scripts, iframes, inline event handlers, or javascript: URLs.
    Returns True if safe, False if unsafe.
    """
    if not isinstance(raw_html, str):
        return False

    decoded_html = raw_html
    for _ in range(3):
        unescaped = html.unescape(decoded_html)
        if unescaped == decoded_html:
            break
        decoded_html = unescaped

    unquoted_html = urllib.parse.unquote(decoded_html)

    # 1. Script / Iframe tag check via regex
    tag_regex = re.compile(r"<\s*/?\s*(script|iframe)(?:[\s/>\x00]|$)", re.IGNORECASE)
    if tag_regex.search(raw_html) or tag_regex.search(decoded_html) or tag_regex.search(unquoted_html):
        return False

    # 2. Inline event handlers
    event_handler_regex = re.compile(
        r"<\s*[a-zA-Z0-9_\-]+[^>]*?[\s/\x00]on[a-zA-Z]+\s*=",
        re.IGNORECASE | re.DOTALL,
    )
    if event_handler_regex.search(raw_html) or event_handler_regex.search(decoded_html) or event_handler_regex.search(unquoted_html):
        return False

    loose_event_regex = re.compile(
        r"<\s*[a-zA-Z0-9_\-]+[^>]*?[\s/\x00]on[a-zA-Z]+(?:\s*=\s*['\"a-zA-Z0-9_\(\)]|\s*>|\s*$)",
        re.IGNORECASE | re.DOTALL,
    )
    if loose_event_regex.search(raw_html) or loose_event_regex.search(decoded_html):
        return False

    # 3. javascript: URLs
    def strip_invisible(s: str) -> str:
        return re.sub(r"[\s\x00-\x1f\x7f-\x9f]", "", s)

    clean_raw = strip_invisible(raw_html.lower())
    clean_decoded = strip_invisible(decoded_html.lower())
    clean_unquoted = strip_invisible(unquoted_html.lower())

    if "javascript:" in clean_raw or "javascript:" in clean_decoded or "javascript:" in clean_unquoted:
        return False

    if re.search(r"javascript\s*(?::|&#58;|&colon;|&#x3a;)", raw_html, re.IGNORECASE) or \
       re.search(r"javascript\s*(?::|&#58;|&colon;|&#x3a;)", decoded_html, re.IGNORECASE):
        return False

    # 4. DOM Parser verification
    try:
        parser = SafeHTMLParser()
        parser.feed(raw_html)
        if parser.is_unsafe:
            return False
    except Exception:
        if "<" in raw_html:
            return False

    return True


def is_exact_email_domain(to_str: str, expected_domain: str) -> bool:
    """
    Validates that recipient address matches expected domain exactly.
    """
    if not isinstance(to_str, str):
        return False

    to_str = to_str.strip()
    if not to_str:
        return False

    if any(c in to_str for c in ["\r", "\n", ",", ";"]):
        return False

    if "<" in to_str and ">" in to_str:
        match = re.match(r"^([^<>\r\n]*?)<([^\s<>]+)>$", to_str)
        if match:
            to_str = match.group(2).strip()
        else:
            return False
    elif " " in to_str or "\t" in to_str:
        return False

    if to_str.count("@") != 1:
        return False

    local_part, domain = to_str.split("@", 1)
    local_part = local_part.strip()
    domain = domain.strip().lower()

    if not local_part or not domain:
        return False

    return domain == expected_domain.lower()


def evaluate_action_firewall(
    payload: Any,
    assigned_tenant: str = ASSIGNED_TENANT,
    assigned_email_domain: str = ASSIGNED_EMAIL_DOMAIN,
) -> Dict[str, str]:
    """
    Evaluates proposed LLM action against deterministic firewall policies.
    """
    # 1. Top-Level Schema Check -> INVALID_SCHEMA
    if not isinstance(payload, dict):
        return {"decision": "block", "reason": "INVALID_SCHEMA"}

    allowed_top_keys = {"provenance", "humanApproved", "untrustedContent", "action"}
    if not set(payload.keys()).issubset(allowed_top_keys):
        return {"decision": "block", "reason": "INVALID_SCHEMA"}

    if "provenance" not in payload or "humanApproved" not in payload or "action" not in payload:
        return {"decision": "block", "reason": "INVALID_SCHEMA"}

    provenance = payload["provenance"]
    if not isinstance(provenance, str) or provenance not in ("trusted", "untrusted"):
        return {"decision": "block", "reason": "INVALID_SCHEMA"}

    human_approved = payload["humanApproved"]
    if type(human_approved) is not bool:
        return {"decision": "block", "reason": "INVALID_SCHEMA"}

    if "untrustedContent" in payload and payload["untrustedContent"] is not None:
        if not isinstance(payload["untrustedContent"], str):
            return {"decision": "block", "reason": "INVALID_SCHEMA"}

    action = payload["action"]
    if not isinstance(action, dict):
        return {"decision": "block", "reason": "INVALID_SCHEMA"}

    if set(action.keys()) != {"tool", "args"}:
        return {"decision": "block", "reason": "INVALID_SCHEMA"}

    tool = action["tool"]
    args = action["args"]

    if not isinstance(tool, str) or not isinstance(args, dict):
        return {"decision": "block", "reason": "INVALID_SCHEMA"}

    # 2. Tool Allowlist Check -> TOOL_NOT_ALLOWED
    if tool not in ALLOWED_TOOLS:
        return {"decision": "block", "reason": "TOOL_NOT_ALLOWED"}

    # 3. Selected Tool's Argument Schema Check -> INVALID_SCHEMA
    if tool == "search":
        if set(args.keys()) != {"query"}:
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        query = args["query"]
        if not isinstance(query, str) or not (1 <= len(query) <= 200):
            return {"decision": "block", "reason": "INVALID_SCHEMA"}

    elif tool == "lookup_record":
        if set(args.keys()) != {"tenantId", "recordId"}:
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        tenant_id = args["tenantId"]
        record_id = args["recordId"]
        if not isinstance(tenant_id, str) or not isinstance(record_id, str) or len(record_id) == 0:
            return {"decision": "block", "reason": "INVALID_SCHEMA"}

    elif tool == "send_email":
        if set(args.keys()) != {"to", "subject", "body"}:
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        to_val = args["to"]
        subject_val = args["subject"]
        body_val = args["body"]
        if not isinstance(to_val, str) or not isinstance(subject_val, str) or not isinstance(body_val, str):
            return {"decision": "block", "reason": "INVALID_SCHEMA"}

    elif tool == "render_html":
        if set(args.keys()) != {"html"}:
            return {"decision": "block", "reason": "INVALID_SCHEMA"}
        html_val = args["html"]
        if not isinstance(html_val, str):
            return {"decision": "block", "reason": "INVALID_SCHEMA"}

    # 4. Tenant Scope Check -> TENANT_SCOPE
    if tool == "lookup_record":
        if args["tenantId"] != assigned_tenant:
            return {"decision": "block", "reason": "TENANT_SCOPE"}

    # 5. Exact Email Domain Check -> EGRESS_DENIED
    if tool == "send_email":
        if not is_exact_email_domain(args["to"], assigned_email_domain):
            return {"decision": "block", "reason": "EGRESS_DENIED"}

    # 6. Human Approval Check -> APPROVAL_REQUIRED
    if tool == "send_email":
        if not human_approved:
            return {"decision": "block", "reason": "APPROVAL_REQUIRED"}

    # 7. HTML Safety Check -> UNSAFE_OUTPUT
    if tool == "render_html":
        if not check_html_safety(args["html"]):
            return {"decision": "block", "reason": "UNSAFE_OUTPUT"}

    # All rules passed -> ALLOW
    return {"decision": "allow", "reason": "ALLOW"}
