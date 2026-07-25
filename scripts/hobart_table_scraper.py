#!/usr/bin/env python3
"""Scrape the DSRG parameter tables from Brouwer & Hobart's page into a CSV.

Source: https://homepages.cwi.nl/~aeb/math/dsrg/dsrg.html

The page lists directed strongly regular graph parameter sets in complementary
pairs: the first row of a pair carries v, the second leaves the v cell empty.
Since a dsrg exists iff its complement does, each pair collapses to one CSV row,
using the parameters of the smaller-k member. The existence verdict is taken
from whichever member of the pair carries a comment.

Comment conventions on the page:
  "?"                    -> existence open
  "does not exist by X"  -> nonexistence proof
  anything else (T*/M*)  -> a construction, so the graph is known
  empty                  -> no information in that row (look at the partner)

Usage: python3 hobart_open_table_scraper.py [-o dsrg.csv] [--html cached.html]
"""

import argparse
import csv
import html
import re
import sys
import urllib.request

URL = "https://homepages.cwi.nl/~aeb/math/dsrg/dsrg.html"

ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S | re.I)
CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")


def fetch(path=None):
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    with urllib.request.urlopen(URL) as r:
        return r.read().decode("utf-8")


def clean(cell):
    """Strip markup from a table cell and normalise whitespace."""
    text = TAG_RE.sub(" ", cell)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def classify(comment):
    """Map one comment cell to 'impossible' / 'open' / 'known' / None."""
    if not comment:
        return None
    if "does not exist" in comment:
        return "impossible"
    if comment == "?":
        return "open"
    return "known"


def combine(a, b):
    """Merge the verdicts of the two members of a complementary pair."""
    verdicts = {v for v in (a, b) if v}
    if not verdicts:
        return None
    # A nonexistence proof beats everything; a construction beats "?".
    for level in ("impossible", "known", "open"):
        if level in verdicts:
            return level


def parse(page):
    """Yield (n, k, t, lam, mu, status) tuples, one per complementary pair."""
    pending = None  # first row of the current pair
    current_v = None
    for raw in ROW_RE.findall(page):
        cells = [clean(c) for c in CELL_RE.findall(raw)]
        if len(cells) != 8:
            continue  # header rows and any stray markup
        v, k, t, lam, mu, comment = *cells[:5], cells[7]  # cells 5,6 are r^f, s^g
        if v:
            if pending:
                yield emit(pending, None)
                pending = None
            current_v = int(v)
        params = (current_v, int(k), int(t), int(lam), int(mu))
        verdict = classify(comment)
        if pending is None:
            pending = (params, verdict)
        else:
            yield emit(pending, verdict)
            pending = None
    if pending:
        yield emit(pending, None)


def emit(pending, partner_verdict):
    params, verdict = pending
    status = combine(verdict, partner_verdict)
    if status is None:
        # No comment on either member: the page offers no construction and no
        # nonexistence proof, which is what "open" means here.
        status = "open"
    return params + (status,)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="dsrg.csv", help="output CSV path")
    ap.add_argument("--html", help="use a local copy instead of fetching")
    args = ap.parse_args()

    rows = list(parse(fetch(args.html)))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "k", "t", "lambda", "mu", "status"])
        w.writerows(rows)

    counts = {}
    for row in rows:
        counts[row[5]] = counts.get(row[5], 0) + 1
    print(f"{len(rows)} parameter sets -> {args.out}", file=sys.stderr)
    for status in ("known", "open", "impossible"):
        print(f"  {status}: {counts.get(status, 0)}", file=sys.stderr)


if __name__ == "__main__":
    main()
