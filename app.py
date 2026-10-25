"""Optional server for Journalist Aws, so visitors need no OpenAI key. NOT DEPLOYED yet: Render is suspended until 2026-11-01.

When deployed (a Render web service, env var OPENAI_API_KEY), put its URL in site/data/config.json as
    { "aiProxy": "https://<service>.onrender.com/api/aws" }
and the site uses it for everyone who has not pasted or paired their own key.

It only forwards a chat request to OpenAI after checking it: Aws prompts only, allowed model, capped size and tokens, per-visitor rate
limit, a global daily budget (so the owner's credit cannot be drained), and CORS only for the World Watch site. Identical requests
(the same world briefing asked by many visitors) are answered from a 10-minute cache at no cost. The key never leaves the server and
no prompt or answer is ever logged.

    python tools/ai_proxy.py          # local test on port 5225
    python -m unittest tools/test_ai_proxy.py
"""
import collections, hashlib, json, os, time
from flask import Flask, request, Response, jsonify

SITE = os.environ.get("WW_SITE", "https://death242-worldwatch.static.hf.space")
ALLOWED_MODELS = {"gpt-5-mini"}
MAX_CHARS, MAX_TOKENS, PER_MINUTE, PER_DAY = 40_000, 1000, 8, 120
DAILY_BUDGET = int(os.environ.get("AWS_DAILY_BUDGET", "400"))          # upstream OpenAI calls per UTC day, all visitors together
CACHE_TTL, CACHE_MAX = 600, 60
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
VERSION = "2"
STARTED = time.time()

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
    try:
        tokens = min(int(body.get("max_completion_tokens") or MAX_TOKENS), MAX_TOKENS)
    except (TypeError, ValueError):
        return None, "bad token limit"
    return {"model": model, "stream": True, "max_completion_tokens": max(1, tokens), "reasoning_effort": "minimal", "messages": [{"role": m["role"], "content": m["content"]} for m in msgs]}, None


def rate_limited(who, now=None):
    now = now or time.time()
    h = hits[who] = [t for t in hits[who] if now - t < 86400]
    if len([t for t in h if now - t < 60]) >= PER_MINUTE or len(h) >= PER_DAY:
        return True
    h.append(now)
    return False


# ---------- response cache: the same question within 10 minutes is answered once ----------
class Cache:
    def __init__(self, ttl=CACHE_TTL, maxsize=CACHE_MAX):
        self.ttl, self.maxsize, self.data = ttl, maxsize, collections.OrderedDict()

    def key(self, body):
        return hashlib.sha256(json.dumps([body["model"], body["messages"]], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, key, now=None):
        now = now or time.time()
        hit = self.data.get(key)
        if not hit or now - hit[0] > self.ttl:
            self.data.pop(key, None)
            return None
        self.data.move_to_end(key)
        return hit[1]

    def put(self, key, text, now=None):
        if not text or not text.strip():
            return
        self.data[key] = (now or time.time(), text)
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
    for i in range(0, len(text), size):
        yield ("data: " + json.dumps({"choices": [{"delta": {"content": text[i:i + size]}}]}) + "\n\n").encode()
    yield b"data: [DONE]\n\n"


# ---------- daily budget and usage counters (no content, only numbers) ----------
class Budget:
    def __init__(self, limit=DAILY_BUDGET):
        self.limit, self.day, self.counts = limit, "", collections.Counter()

    def _roll(self, now):
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self.day:
            self.day, self.counts = day, collections.Counter()

    def try_spend(self, now=None):
        self._roll(now or time.time())
        if self.counts["upstream"] >= self.limit:
            self.counts["budget_refused"] += 1
            return False
        self.counts["upstream"] += 1
        return True

    def note(self, what, now=None):
        self._roll(now or time.time())
        self.counts[what] += 1

    def today(self, now=None):
        self._roll(now or time.time())
        return {"date": self.day, "budget": self.limit, **{k: self.counts[k] for k in ("requests", "upstream", "cache_hits", "rate_limited", "refused", "budget_refused", "errors")}}


cache, budget = Cache(), Budget()


@app.route("/health")
def health():                                    # the site pings this when it opens, so Render's free plan is already awake when someone asks Aws
    return cors(jsonify(ok=True, version=VERSION, uptime_s=int(time.time() - STARTED), today=budget.today()))


@app.route("/api/aws", methods=["POST", "OPTIONS"])
def aws():
    if request.method == "OPTIONS":
        return cors(Response(status=204))
    import requests
    budget.note("requests")
    body, err = check_request(request.get_json(silent=True))
    if err:
        budget.note("refused")
        return cors(jsonify(error={"message": err})), 400
    key = cache.key(body)
    cached = cache.get(key)
    if cached is not None:                                                       # free: nobody is charged for a repeat question
        budget.note("cache_hits")
        resp = Response(replay_sse(cached), content_type="text/event-stream")
        resp.headers["X-Cache"] = "HIT"
        return cors(resp)
    who = (request.headers.get("X-Forwarded-For", request.remote_addr) or "?").split(",")[0].strip()
    if rate_limited(who):
        budget.note("rate_limited")
        return cors(jsonify(error={"message": "slow down"})), 429
    secret = os.environ.get("OPENAI_API_KEY")
    if not secret:
        return cors(jsonify(error={"message": "not configured"})), 503
    if not budget.try_spend():
        return cors(jsonify(error={"message": "Aws has reached today's limit. Try again tomorrow, or use your own OpenAI key in Aws's settings."})), 429
    try:
        up = requests.post(OPENAI_URL, headers={"Authorization": "Bearer " + secret}, json=body, stream=True, timeout=(6, 40))
    except requests.RequestException:
        budget.note("errors")
        return cors(jsonify(error={"message": "upstream unavailable"})), 502

    def relay():
        lines = []
        for raw in up.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            lines.append(raw)
            yield (raw + "\n").encode()
        if up.status_code == 200:
            cache.put(key, sse_text(lines))
        else:
            budget.note("errors")

    resp = Response(relay(), status=up.status_code, content_type=up.headers.get("Content-Type", "text/event-stream"))
    resp.headers["X-Cache"] = "MISS"
    return cors(resp)


if __name__ == "__main__":
    app.run(port=5225)
