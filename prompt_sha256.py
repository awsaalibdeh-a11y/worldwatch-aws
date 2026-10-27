"""Prints the SHA-256 values the proxy compares against: one per Aws mode (briefing, place, chat).

Only the fixed instructions are hashed (everything before "=== DATA (untrusted) ==="), because the site adds a live
data block that changes on every request. Run it from the World Watch folder; it asks the site's own code for the
prompts, so the hashes always match what the site sends:

    python deploy/aws-proxy/prompt_sha256.py            # needs node; reads site/js/core/aws.js
    python prompt_sha256.py prompt.txt                  # or hash one captured full prompt

Put the comma-separated result in the Render environment as AWS_PROMPT_SHA256, and run it again whenever the
instructions in site/js/core/aws.js change.
"""
import hashlib
import os
import subprocess
import sys

OPEN = "\n=== DATA (untrusted) ===\n"


def head_hash(text):
    head, sep, _ = text.partition(OPEN)
    if not sep:
        sys.exit("that is not a full Aws prompt (no DATA block)")
    return hashlib.sha256(head.encode()).hexdigest()


def from_site():
    here = os.path.dirname(os.path.abspath(__file__))
    aws = os.path.join(here, "..", "..", "site", "js", "core", "aws.js")
    if not os.path.exists(aws):
        sys.exit("site/js/core/aws.js not found: run this from the World Watch folder, or pass a prompt file")
    url = "file:///" + os.path.abspath(aws).replace("\\", "/")
    js = ("import(%r).then(m => console.log(JSON.stringify(['briefing', 'place', 'chat']"
          ".map(k => m.systemPrompt(k, 'x')))))" % url)
    out = subprocess.run(["node", "--input-type=module", "-e", js], capture_output=True, check=True).stdout.decode("utf-8")          # node prints UTF-8; Windows would guess cp1252
    import json
    return [head_hash(p) for p in json.loads(out)]


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as fh:
            print(head_hash(fh.read()))
    else:
        print(",".join(from_site()))


if __name__ == "__main__":
    main()
