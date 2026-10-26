"""Prints the SHA-256 the proxy compares against, for the Aws system prompt the site sends.

Usage:
    python prompt_sha256.py                 # reads the prompt from stdin
    python prompt_sha256.py prompt.txt      # or from a file

Put the value in the Render environment as AWS_PROMPT_SHA256. The site and the proxy must send a
byte-identical system prompt, so keep the text in one place on the site and generate the hash from it.
"""
import hashlib
import sys


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()
    if not text:
        sys.exit("empty prompt")
    print(hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
