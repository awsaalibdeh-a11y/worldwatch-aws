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

# The site's system prompt is not one fixed string. It is a fixed instruction block followed by a live
# briefing, and the live part changes on every request (map data, task choice). Hashing the whole message
# therefore refused every real call. Only the fixed block is fingerprinted.
#
#   <fixed instructions>            <- fingerprinted, must equal one of AWS_PROMPT_SHA256
#   === DATA (untrusted) ===        <- anything after this line is ignored
#   <live map / world data>
#
# The marker is the split between trusted text and untrusted data, so it must appear exactly once and only
# the text before it is checked.
DATA_MARKER = "=== DATA (untrusted) ==="
# Accepted fingerprints: comma-separated hex digests, one per task variant (briefing / place / chat).
AWS_PROMPT_SHA256S = tuple(h.strip().lower() for h in os.environ.get("AWS_PROMPT_SHA256", "").split(",") if h.strip())
# Kept as an alias so older deploys and the tests that read it keep working.
AWS_PROMPT_SHA256 = AWS_PROMPT_SHA256S[0] if AWS_PROMPT_SHA256S else ""

COUNTER_KEYS = ("requests", "upstream", "cache_hits", "rate_limited", "refused", "budget_refused", "errors")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aws_proxy")

app = Flask(__name__)
hits = collections.defaultdict(list)
hits_lock = threading.Lock()

if not AWS_PROMPT_SHA256S:
    log.warning("AWS_PROMPT_SHA256 is not set: falling back to the prefix check, which is not a real allow-list")
else:
    log.info("AWS_PROMPT_SHA256 accepts %d prompt variant(s)", len(AWS_PROMPT_SHA256S))


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


# Which X-Forwarded-For entry is the visitor. This CANNOT be settled by reasoning: it depends on how many
# proxies sit in front of the app and what each one appends. The wrong choice is not loud — it silently
# gives every visitor the same identity, so they all share one rate limit.
#
#   "right" (default)  the last entry: what the nearest edge appended. Correct for Render, whose proxy
#                      appends the peer address and strips a caller-supplied chain.
#   "left"             the first entry: the address the original caller claimed. Correct when the app sits
#                      behind a chain that preserves the client's own value at the head.
#   "remote"           ignore the header entirely and use the socket peer.
#
# Check it on the live proxy with GET /health/ip while sending a known X-Forwarded-For. If the reported
# address is Render's own load balancer for every visitor, switch this to "remote" or "left".
IP_SOURCE = os.environ.get("AWS_IP_SOURCE", "right").strip().lower()


@app.errorhandler(Exception)
def on_error(exc):
    """Never let an unhandled error reach the browser as a bare HTML 500.

    HTTP errors (404 for an unknown path, 405 for a wrong method) are normal answers, not faults. Letting
    them fall through to this handler turned every bot probe into a 500 and wrote a full traceback to the
    log, which both inflated the error counter and buried real faults. Answer them plainly instead.
    """
    if isinstance(exc, HTTPException):
        return jsonify(error={"message": exc.description}), exc.code
    budget.note("errors")
    log.exception("unhandled error on %s", request.path)
    return jsonify(error={"message": "server error"}), 500


def fixed_part(text):
    """The trusted instruction block: everything before the data marker, with trailing whitespace trimmed.

    Returns None when the marker is absent or appears more than once. More than one occurrence means the
    caller is trying to hide extra instructions after a second marker, so the message is refused.
    """
    if text.count(DATA_MARKER) != 1:
        return None
    return text.split(DATA_MARKER, 1)[0].strip()


def is_aws_prompt(text):
    """True when the trusted part of the system message is one of the accepted Aws prompt variants."""
    head = fixed_part(text)
    if head is None:
        return False
    if AWS_PROMPT_SHA256S:
        digest = hashlib.sha256(head.encode()).hexdigest()
        return digest in AWS_PROMPT_SHA256S
    return head.startswith(AWS_SYSTEM_PREFIX)


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


def forwarded_chain():
    """Every X-Forwarded-For entry, left to right, as the caller sent them."""
    raw = request.headers.get("X-Forwarded-For", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def client_ip():
    """The visitor's address, by the policy in IP_SOURCE.

    This is the only place the choice is made, so the rate limiter and the diagnostic endpoint can never
    disagree. See the comment on IP_SOURCE for why the right answer is a deployment fact, not a guess.
    """
    chain = forwarded_chain()
    if IP_SOURCE == "remote" or not chain:
        return request.remote_addr or "?"
    if IP_SOURCE == "left":
        return chain[0]
    return chain[-1]


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


@app.route("/health/ip", methods=["GET", "OPTIONS"])
def health_ip():
    """Shows how the proxy identifies the caller, so the X-Forwarded-For policy can be checked on the live
    service instead of assumed. Behind AWS_ADMIN_TOKEN: it reveals the shape of your edge, not just a number.

    Open it twice from two different networks (phone on mobile data, laptop on wifi). If `chosen` is the
    same for both, every visitor is sharing one rate limit and AWS_IP_SOURCE needs setting.
    """
    if request.method == "OPTIONS":
        return Response(status=204)
    admin = os.environ.get("AWS_ADMIN_TOKEN")
    if not admin or request.headers.get("X-Admin-Token") != admin:
        return jsonify(error={"message": "not found"}), 404
    payload = {"source": IP_SOURCE, "chain": forwarded_chain(),
               "remote_addr": request.remote_addr, "chosen": client_ip()}
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


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def not_found(path):
    return jsonify(error={"message": "not found"}), 404


if __name__ == "__main__":
    app.run(port=5225)
