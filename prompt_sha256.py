"""Prints the fingerprint(s) the proxy expects for the Aws system prompt.

The site sends a fixed instruction block followed by live data, separated by the DATA marker:

    You are Aws, ...
    <the fixed instructions for this task>
    === DATA (untrusted) ===
    <map / world data, changes every request>

Only the part before the marker is trusted, so only that part is fingerprinted. Run this on each task
variant (briefing, place, chat) and put all the digests in one comma-separated value:

    export AWS_PROMPT_SHA256="<briefing>,<place>,<chat>"

Usage:
    python prompt_sha256.py                     # a whole system message on stdin
    python prompt_sha256.py prompt.txt          # or from a file
    python prompt_sha256.py a.txt b.txt c.txt   # one digest per line, in order

The marker line and anything after it are ignored, so pasting a captured request works directly.
"""
import hashlib
import sys

MARKER = "=== DATA (untrusted) ==="


def digest(text):
    """Fingerprint of the trusted part: everything before the marker, trailing whitespace trimmed."""
    if MARKER in text:
        text = text.split(MARKER, 1)[0]
    head = text.strip()
    if not head:
        raise SystemExit("empty prompt (nothing before the data marker)")
    return hashlib.sha256(head.encode()).hexdigest()


def main():
    paths = sys.argv[1:]
    if not paths:
        print(digest(sys.stdin.read()))
        return
    digests = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            digests.append(digest(fh.read()))
    print(",".join(digests))


if __name__ == "__main__":
    main()
