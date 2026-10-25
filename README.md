# World Watch: Aws proxy

A tiny Flask server for Journalist Aws (the AI correspondent in World Watch, made by Awsaa). It holds the OpenAI key (env var `OPENAI_API_KEY`) so visitors need no key, and forwards only checked requests:
Aws prompts only, model gpt-5-mini, size and token caps, per-visitor rate limits, CORS only for the World Watch site.

- `POST /api/aws`: streamed chat completion
- `GET /health`: wakes the free plan
