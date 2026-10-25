"""Optional server for Journalist Aws, so visitors need no OpenAI key. NOT DEPLOYED: Render is suspended until 2026-11-01.

When deployed (a Render web service, env var OPENAI_API_KEY), put its URL in site/data/config.json as
    { "aiProxy": "https://<service>.onrender.com/api/aws" }
and the site uses it for everyone who has not pasted their own key.

It only forwards a chat request to OpenAI after checking it: allowed model, capped size and tokens, per-visitor rate limit,
and CORS only for the World Watch site. The key never leaves the server.

    python tools/ai_proxy.py          # local test on port 5225
    python -m unittest tools/test_ai_proxy.py
"""
import os, time, json, collections
from flask import Flask, request, Response, jsonify

SITE = os.environ.get("WW_SITE", "https://death242-worldwatch.static.hf.space")
ALLOWED_MODELS = {"gpt-5-mini"}
MAX_CHARS, MAX_TOKENS, PER_MINUTE, PER_DAY = 40_000, 1000, 8, 120
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

app = Flask(__name__)
hits = collections.defaultdict(list)


def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = SITE
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Vary"] = "Origin"
    return resp


def check_request(body):
    """Returns (cleaned body, None) or (None, error message)."""
    if not isinstance(body, dict):
        return None, "bad request"
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not 1 <= len(msgs) <= 12:
        return None, "messages must be a list of 1-12"
    total = 0
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
            return None, "bad message"
        total += len(m["content"])
    if total > MAX_CHARS:
        return None, "too long"
    if msgs[0]["role"] != "system" or not msgs[0]["content"].startswith("You are Aws"):
        return None, "this proxy only serves Journalist Aws"
    model = body.get("model", "gpt-5-mini")
    if model not in ALLOWED_MODELS:
        return None, "model not allowed"
    return {"model": model, "stream": True, "max_completion_tokens": min(int(body.get("max_completion_tokens") or MAX_TOKENS), MAX_TOKENS), "reasoning_effort": "minimal", "messages": msgs}, None


def rate_limited(who, now=None):
    now = now or time.time()
    h = hits[who] = [t for t in hits[who] if now - t < 86400]
    if len([t for t in h if now - t < 60]) >= PER_MINUTE or len(h) >= PER_DAY:
        return True
    h.append(now)
    return False


@app.route("/health")
def health():                                    # the site pings this when it opens, so Render's free plan is already awake when someone asks Aws
    return cors(Response("ok", mimetype="text/plain"))


@app.route("/api/aws", methods=["POST", "OPTIONS"])
def aws():
    if request.method == "OPTIONS":
        return cors(Response(status=204))
    import requests
    body, err = check_request(request.get_json(silent=True))
    if err:
        return cors(jsonify(error={"message": err})), 400
    who = (request.headers.get("X-Forwarded-For", request.remote_addr) or "?").split(",")[0].strip()
    if rate_limited(who):
        return cors(jsonify(error={"message": "slow down"})), 429
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return cors(jsonify(error={"message": "not configured"})), 503
    up = requests.post(OPENAI_URL, headers={"Authorization": "Bearer " + key}, json=body, stream=True, timeout=(6, 40))
    return cors(Response(up.iter_content(chunk_size=None), status=up.status_code, content_type=up.headers.get("Content-Type", "text/event-stream")))


if __name__ == "__main__":
    app.run(port=5225)
