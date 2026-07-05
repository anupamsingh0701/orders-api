import base64
import math
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

TOTAL_ORDERS = 59
RATE_LIMIT = 20
RATE_WINDOW_SECONDS = 10.0

app = FastAPI(title="Orders API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)

_idempotency_lock = threading.Lock()
_idempotency_store: Dict[str, Dict[str, Any]] = {}

_rate_limit_lock = threading.Lock()
_rate_limit_windows: Dict[str, deque[float]] = defaultdict(deque)

CATALOG: List[Dict[str, Any]] = [
    {
        "id": order_id,
        "sku": f"SKU-{order_id:04d}",
        "item": f"Order {order_id}",
        "status": "ready",
    }
    for order_id in range(1, TOTAL_ORDERS + 1)
]


def _encode_cursor(next_id: int) -> str:
    return base64.urlsafe_b64encode(f"order:{next_id}".encode("utf-8")).decode("ascii")


def _decode_cursor(cursor: str) -> int:
    try:
        raw_value = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        prefix, value = raw_value.split(":", 1)
        if prefix != "order":
            raise ValueError("bad prefix")
        next_id = int(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid cursor") from exc

    if next_id < 1 or next_id > TOTAL_ORDERS + 1:
        raise HTTPException(status_code=400, detail="Invalid cursor")
    return next_id


def _check_rate_limit(client_id: str) -> Optional[int]:
    now = time.time()
    with _rate_limit_lock:
        window = _rate_limit_windows[client_id]
        while window and now - window[0] >= RATE_WINDOW_SECONDS:
            window.popleft()

        if len(window) >= RATE_LIMIT:
            retry_after = max(1, math.ceil(RATE_WINDOW_SECONDS - (now - window[0])))
            return retry_after

        window.append(now)
        return None


@app.middleware("http")
async def apply_rate_limit(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    client_id = request.headers.get("X-Client-Id", "").strip() or "anonymous"
    retry_after = _check_rate_limit(client_id)
    if retry_after is not None:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too Many Requests"},
            headers={"Retry-After": str(retry_after)},
        )

    return await call_next(request)


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "orders-api",
        "patterns": ["idempotent-post", "cursor-pagination", "per-client-rate-limit"],
        "total_orders": TOTAL_ORDERS,
        "rate_limit": f"{RATE_LIMIT}/10s",
    }


@app.post("/orders", status_code=201)
async def create_order(
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
) -> JSONResponse:
    key = idempotency_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Idempotency-Key header is required")

    with _idempotency_lock:
        existing = _idempotency_store.get(key)
        if existing is not None:
            return JSONResponse(status_code=201, content=existing)

        payload = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        order = {
            "id": str(uuid.uuid4()),
            "item": payload.get("item", "Sample Item"),
            "quantity": payload.get("quantity", 1),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _idempotency_store[key] = order
        return JSONResponse(status_code=201, content=order)


@app.get("/orders")
async def list_orders(limit: int = 10, cursor: Optional[str] = None) -> Dict[str, Any]:
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be positive")

    start_id = _decode_cursor(cursor) if cursor else 1
    start_index = start_id - 1
    page_items = CATALOG[start_index : start_index + limit]

    if not page_items:
        return {"items": [], "next_cursor": None}

    last_id = page_items[-1]["id"]
    next_cursor = _encode_cursor(last_id + 1) if last_id < TOTAL_ORDERS else None
    return {"items": page_items, "next_cursor": next_cursor}
