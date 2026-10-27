"""Optional server for Journalist Aws, so visitors need no OpenAI key.

When deployed (a Render web service, env var OPENAI_API_KEY), put its URL in site/data/config.json as
    { "aiProxy": "https://<service>.onrender.com/api/aws" }
and the site uses it for everyone who has not pasted or paired their own key.

It only forwards a chat request to OpenAI after checking it: an allow-listed Aws system prompt (matched by
SHA-256, not by prefix), allowed model, capped size and tokens, per-visitor rate limit keyed on the real
client address, a global daily budget (so the owner's credit cannot be drained), and CORS only for the
World Watch site. Identical requests are answered from a 10-minute cache at no cost. The key never leaves
the server, no prompt or answer is ever logged, and upstream error bodies are never relayed to the browser.

    python app.py                        # local test on port 5225
    python -m unittest test_app.py       # run the tests

Required environment:
    OPENAI_API_KEY   the key the proxy spends
    AWS_PROMPT_SHA256  hex SHA-256 of the exact Aws system prompt (see prompt_sha256.py). Optional: if unset,
                       the prefix check is used as a degraded fallback and a warning is logged.
Optional environment:
    WW_SITE            the allowed origin (default: the World Watch site)
    AWS_DAILY_BUDGET   upstream calls allowed per UTC day, all visitors together (default 400)

Single-worker only: the cache, rate limiter and budget live in process memory. Run gunicorn with
--workers 1 (as render.yaml does) or move them to Redis.
"""
import collections, hashlib, json, logging, os, threading, time

import requests
from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException

SITE = os.environ.get("WW_SITE", "https://death242-worldwatch.static.hf.space")
ALLOWED_MODELS = {"gpt-5-mini"}
MAX_CHARS, MAX_TOKENS, PER_MINUTE, PER_DAY = 40_000, 1000, 8, 120
DAILY_BUDGET = int(os.environ.get("AWS_DAILY_BUDGET", "400"))          # upstream OpenAI calls per UTC day, all visitors together
CACHE_TTL, CACHE_MAX = 600, 60
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
VERSION = "3"
STARTED = time.time()

AWS_SYSTEM_PREFIX = "You are Aws"
# The site's system prompt is fixed instructions, then a live DATA block that changes on every request:
#     <instructions for briefing / place / chat>\n=== DATA (untrusted) ===\n<live map data>\n=== END DATA ===
# so only the instructions are hashed. AWS_PROMPT_SHA256 is a comma-separated list, one hash per mode.
DATA_OPEN, DATA_CLOSE = "\n=== DATA (untrusted) ===\n", "\n=== END DATA ==="
AWS_PROMPT_SHA256 = os.environ.get("AWS_PROMPT_SHA256", "").strip().lower()

COUNTER_KEYS = ("requests", "upstream", "cache_hits", "rate_limited", "refused", "budget_refused", "errors")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aws_proxy")

app = Flask(__name__)
hits = collections.defaultdict(list)
hits_lock = threading.Lock()

if not AWS_PROMPT_SHA256:
    log.warning("AWS_PROMPT_SHA256 is not set: falling back to the prefix check, which is not a real allow-list")


def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = SITE
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Vary"] = "Origin"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.after_request
def always_cors(resp):
    """CORS on every response, including error handlers and 404s, so the browser shows real statuses."""
    return cors(resp)


@app.errorhandler(Exception)
def on_error(exc):
    """Never let an unhandled error reach the browser as a bare HTML 500.
    Ordinary HTTP errors (404, 405) keep their status and are not counted or logged as server errors."""
    if isinstance(exc, HTTPException):
        return jsonify(error={"message": (exc.name or "error").lower()}), exc.code
    budget.note("errors")
    log.exception("unhandled error on %s", request.path)
    return jsonify(error={"message": "server error"}), 500


def prompt_head(text):
    """The fixed instructions of an Aws prompt (everything before the DATA block), or None if it isn't shaped like one."""
    head, sep, rest = text.partition(DATA_OPEN)
    if not sep or not rest.endswith(DATA_CLOSE):
        return None
    return head


def is_aws_prompt(text):
    """True only for one of the exact Aws instruction sets, whatever the live data block holds."""
    if AWS_PROMPT_SHA256:
        head = prompt_head(text)
        allowed = {h.strip() for h in AWS_PROMPT_SHA256.split(",") if h.strip()}
        return head is not None and hashlib.sha256(head.encode()).hexdigest() in allowed
    return text.startswith(AWS_SYSTEM_PREFIX)


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
    if msgs[0]["role"] != "system" or not is_aws_prompt(msgs[0]["content"]):
        return None, "this proxy only serves Journalist Aws"
    if not any(m["role"] == "user" for m in msgs):
        return None, "bad message"                      # a lone system prompt is not a conversation
    model = body.get("model") or "gpt-5-mini"
    if model not in ALLOWED_MODELS:
        return None, "model not allowed"
    raw_tokens = body.get("max_completion_tokens")
    if raw_tokens is None:
        tokens = MAX_TOKENS
    else:
        try:
            tokens = int(raw_tokens)
        except (TypeError, ValueError):
            return None, "bad token limit"
    tokens = max(1, min(tokens, MAX_TOKENS))           # 0 means 1, not "unset"
    return {"model": model, "stream": True, "max_completion_tokens": tokens, "reasoning_effort": "minimal",
            "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]}, None


def client_ip():
    """The real client address.

    On Render the last X-Forwarded-For entry is appended by Render's own edge, so it is the only one
    that cannot be forged by the caller. Earlier entries are attacker-controlled.
    """
    raw = request.headers.get("X-Forwarded-For", "")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if parts:
        return parts[-1]
    return request.remote_addr or "?"


def rate_limited(who, now=None):
    now = time.time() if now is None else now
    with hits_lock:
        h = [t for t in hits.get(who, ()) if now - t < 86400]
        if len([t for t in h if now - t < 60]) >= PER_MINUTE or len(h) >= PER_DAY:
            hits[who] = h
            return True
        h.append(now)
        hits[who] = h
        return False


def prune_hits(now=None):
    """Drop IPs with no activity in the last day so the map cannot grow without bound."""
    now = time.time() if now is None else now
    with hits_lock:
        for who in [k for k, v in hits.items() if not v or now - v[-1] >= 86400]:
            del hits[who]


# ---------- response cache: the same question within 10 minutes is answered once ----------
class Cache:
    def __init__(self, ttl=CACHE_TTL, maxsize=CACHE_MAX):
        self.ttl, self.maxsize, self.data = ttl, maxsize, collections.OrderedDict()
        self.lock = threading.Lock()

    def key(self, body):
        # The token cap is part of the key: a 20-token request must not be served a 1000-token answer.
        return hashlib.sha256(json.dumps([body["model"], body["max_completion_tokens"], body["messages"]],
                                         sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, key, now=None):
        now = time.time() if now is None else now
        with self.lock:
            hit = self.data.get(key)
            if not hit or now - hit[0] > self.ttl:
                self.data.pop(key, None)
                return None
            self.data.move_to_end(key)
            return hit[1]

    def put(self, key, text, now=None):
        if not text or not text.strip():
            return
        with self.lock:
            self.data[key] = (time.time() if now is None else now, text)
            self.data.move_to_end(key)
            while len(self.data) > self.maxsize:
                self.data.popitem(last=False)


def sse_text(lines):
    """Collects the answer text from OpenAI stream lines (str)."""
    out = []
    for ln in lines:
        if not ln.startswith("data:"):
            continue
        d = ln[5:].strip()
        if d == "[DONE]":
            break
        try:
            piece = json.loads(d)["choices"][0]["delta"].get("content")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            continue
        if piece:
            out.append(piece)
    return "".join(out)


def replay_sse(text, size=40):
    """Re-emits a cached answer as a stream the OpenAI client shape accepts (role, id, model, finish)."""
    rid = "chatcmpl-cache-" + hashlib.sha256(text.encode()).hexdigest()[:16]
    created = int(time.time())
    first = True
    for i in range(0, len(text), size):
        delta = {"content": text[i:i + size]}
        if first:
            delta["role"] = "assistant"
            first = False
        yield ("data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                      "model": "gpt-5-mini", "choices": [{"index": 0, "delta": delta,
                                                                          "finish_reason": None}]}) + "\n\n").encode()
    yield ("data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                  "model": "gpt-5-mini",
                                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n").encode()
    yield b"data: [DONE]\n\n"


# ---------- daily budget and usage counters (no content, only numbers) ----------
class Budget:
    def __init__(self, limit=DAILY_BUDGET):
        self.limit, self.day, self.counts = limit, "", collections.Counter()
        self.lock = threading.Lock()

    def _roll(self, now):
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self.day:
            self.day, self.counts = day, collections.Counter()

    def try_spend(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            self._roll(now)
            if self.counts["upstream"] >= self.limit:
                self.counts["budget_refused"] += 1
                return False
            self.counts["upstream"] += 1
            return True

    def note(self, what, now=None):
        now = time.time() if now is None else now
        with self.lock:
            self._roll(now)
            self.counts[what] += 1

    def today(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            self._roll(now)
            return {"date": self.day, "budget": self.limit, **{k: self.counts[k] for k in COUNTER_KEYS}}


cache, budget = Cache(), Budget()


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    """The site pings this when it opens, so Render's free plan is already awake when someone asks Aws.

    Public fields only: the usage counters stay behind a token so visitors cannot read your spend.
    """
    if request.method == "OPTIONS":
        return Response(status=204)
    payload = {"ok": True, "version": VERSION, "uptime_s": int(time.time() - STARTED)}
    admin = os.environ.get("AWS_ADMIN_TOKEN")
    if admin and request.headers.get("X-Admin-Token") == admin:
        payload["today"] = budget.today()
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/aws", methods=["POST", "OPTIONS"])
def aws():
    if request.method == "OPTIONS":
        return Response(status=204)
    budget.note("requests")

    body, err = check_request(request.get_json(silent=True))
    if err:
        budget.note("refused")
        resp = jsonify(error={"message": err})
        resp.headers["Cache-Control"] = "no-store"
        return resp, 400

    key = cache.key(body)
    cached = cache.get(key)
    if cached is not None:                                                       # free: nobody is charged for a repeat question
        budget.note("cache_hits")
        resp = Response(replay_sse(cached), content_type="text/event-stream")
        resp.headers["X-Cache"] = "HIT"
        return resp

    who = client_ip()
    if rate_limited(who):
        budget.note("rate_limited")
        resp = jsonify(error={"message": "slow down"})
        resp.headers["Retry-After"] = "10"
        resp.headers["Cache-Control"] = "no-store"
        return resp, 429

    secret = os.environ.get("OPENAI_API_KEY")
    if not secret:
        return jsonify(error={"message": "not configured"}), 503
    if not budget.try_spend():
        resp = jsonify(error={"message": "Aws has reached today's limit. Try again tomorrow, or use your own OpenAI key in Aws's settings."})
        resp.headers["Cache-Control"] = "no-store"
        return resp, 429

    try:
        up = requests.post(OPENAI_URL, headers={"Authorization": "Bearer " + secret}, json=body,
                           stream=True, timeout=(6, 40))
    except requests.RequestException:
        budget.note("errors")
        log.warning("upstream unreachable")
        return jsonify(error={"message": "upstream unavailable"}), 502

    if up.status_code != 200:
        # Never relay OpenAI's error body: it contains key hints, org ids and quota detail.
        budget.note("errors")
        log.warning("upstream returned %s", up.status_code)
        up.close()
        status = 429 if up.status_code == 429 else 502
        resp = jsonify(error={"message": "Aws is unavailable right now. Try again in a moment."})
        resp.headers["Cache-Control"] = "no-store"
        return resp, status

    def relay():
        lines = []
        try:
            for raw in up.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                lines.append(raw)
                yield (raw + "\n").encode()
        except requests.RequestException:
            budget.note("errors")
            log.warning("upstream stream broke")
            return
        cache.put(key, sse_text(lines))

    resp = Response(relay(), content_type="text/event-stream")
    resp.headers["X-Cache"] = "MISS"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def not_found(path):
    return jsonify(error={"message": "not found"}), 404


if __name__ == "__main__":
    app.run(port=5225)
