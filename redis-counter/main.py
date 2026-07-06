import os
import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Redis Counter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

def get_redis():
    return redis.from_url(REDIS_URL, decode_responses=True)


@app.post("/hit/{key}")
def hit(key: str):
    r = get_redis()
    count = r.incr(key)
    return {"key": key, "count": count}


@app.get("/count/{key}")
def count(key: str):
    r = get_redis()
    val = r.get(key)
    return {"key": key, "count": int(val) if val is not None else 0}


@app.get("/healthz")
def healthz():
    r = get_redis()
    try:
        r.ping()
        redis_status = "up"
    except Exception:
        redis_status = "down"
    return {"status": "ok", "redis": redis_status}
