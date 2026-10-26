# World Watch: Aws proxy

A tiny Flask server for Journalist Aws (the AI correspondent in World Watch, made by Awsaa). It holds the OpenAI key (env var `OPENAI_API_KEY`) so visitors need no key, and forwards only checked requests.

- `POST /api/aws` — streamed chat completion
- `GET /health` — wakes the Render free plan; public fields only

## What it checks before spending your credit

| Check | How |
|---|---|
| Aws prompts only | The **fixed part** of the system prompt (everything before `=== DATA (untrusted) ===`) must match a SHA-256 in `AWS_PROMPT_SHA256`. The live data after the marker is not fingerprinted, and the marker must appear exactly once. A conversation must also contain at least one user turn. |
| Allowed model | `gpt-5-mini` only |
| Size | ≤ 12 messages, ≤ 40 000 characters total |
| Tokens | `max_completion_tokens` clamped to 1–1000 (0 means 1) |
| Rate limit | 8/min and 120/day per real client IP (rightmost `X-Forwarded-For` entry, i.e. the one Render appends) |
| Daily budget | 400 upstream calls per UTC day, all visitors together (`AWS_DAILY_BUDGET`) |
| CORS | Only `WW_SITE`, on every response including errors |

Identical requests (the same world briefing asked by many visitors) are answered from a 10-minute cache at no cost. The key never leaves the server, no prompt or answer is ever logged, and **upstream error bodies are never relayed to the browser** — visitors get a generic "Aws is unavailable" instead of OpenAI's key/quota messages.

## Environment

Required:

- `OPENAI_API_KEY` — the key the proxy spends.
- `AWS_PROMPT_SHA256` — comma-separated hex SHA-256 digests, one per Aws task variant (briefing, place, chat). Only the text before `=== DATA (untrusted) ===` is hashed, so the changing map data does not affect it. Generate all of them at once with `python prompt_sha256.py briefing.txt place.txt chat.txt`, or pipe a captured system message into `python prompt_sha256.py`. If unset, the proxy logs a warning and falls back to the old prefix check, which is **not** a real allow-list. If set, it must contain every variant the site sends, or the missing ones are refused.

Optional:

- `WW_SITE` — the allowed origin (default `https://death242-worldwatch.static.hf.space`).
- `AWS_DAILY_BUDGET` — upstream calls per UTC day (default `400`).
- `AWS_ADMIN_TOKEN` — if set, `GET /health` with header `X-Admin-Token: <token>` also returns usage counters. Without it they are not exposed.

## Run

```bash
pip install -r requirements.txt
python app.py                     # local, port 5225
python -m unittest test_app.py    # tests
```

Production, on Render, uses `--workers 1` **on purpose**: the cache, rate limiter and budget are in process memory. Running more than one worker splits the budget across processes and multiplies your real spend ceiling. To scale out, move those three to Redis first.

## Deploy

See `render.yaml`. After the service is up, put its URL plus `/api/aws` in `site/data/config.json` as `aiProxy`.

## Known limits

- State is in memory: a restart clears the cache, the rate-limit windows and the daily counter. The daily counter therefore resets on each deploy/restart — keep the budget slightly below what you can absorb.
- Rate limiting by IP is a blunt tool; shared NATs (schools, offices) share one allowance.
