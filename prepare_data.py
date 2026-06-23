"""
pull down a public-domain french text (voltaire's candide) and chop off the
project gutenberg header/footer so we're left with just the actual book.

    python prepare_data.py   -> writes data/corpus.txt

swap GUTENBERG_URL for any other plain-text book if you want a different style.
"""

import os
import re
import urllib.request

GUTENBERG_URL = "https://www.gutenberg.org/files/4650/4650-0.txt"
OUT = "data/corpus.txt"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "minigpt/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def strip_gutenberg(text: str) -> str:
    """keep only what's between the START and END markers gutenberg adds."""
    start = re.search(r"\*\*\* START OF.*?\*\*\*", text, re.S)
    end = re.search(r"\*\*\* END OF.*?\*\*\*", text, re.S)
    s = start.end() if start else 0
    e = end.start() if end else len(text)
    body = text[s:e]
    body = re.sub(r"\n{3,}", "\n\n", body).strip()   # squash big gaps of blank lines
    return body


def main():
    os.makedirs("data", exist_ok=True)
    print(f"downloading {GUTENBERG_URL} ...")
    raw = fetch(GUTENBERG_URL)
    corpus = strip_gutenberg(raw)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"wrote {OUT}  ({len(corpus):,} chars, {len(set(corpus))} unique)")


if __name__ == "__main__":
    main()
