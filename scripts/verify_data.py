#!/usr/bin/env python3
"""Verify the integrity of a source-of-truth data dir.

The data dir must hold `searches.csv` and a `dpds/` directory of `nNNN.csv`
shards (see data2/ for an example). Three checks are run:

  1. Every dpds construction really is a DSRG with its stated (n,k,t,lambda,mu).
  2. Every dpds construction lists its members in strictly ascending order.
  3. searches.num_dpds lines up with the dpds constructions: for each search
     method, the count equals the number of constructions of that (group,param)
     whose source_method includes the method.

Exits nonzero if any check fails. Usage: verify_data.py <data_dir>.
"""

import argparse
import collections
from collections.abc import Callable
import csv
import glob
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import TypedDict

from tqdm import tqdm

# Methods recorded in searches.csv; other dpds source_methods (e.g. `inverse`)
# are constructions derived post-hoc and are not tracked as searches.
SEARCH_METHODS: set[str] = {"exhaustive", "ilp", "rrhc"}


def is_search_method(m: str) -> bool:
    """Budget-tagged methods (ilp-30, ilp-720, rrhc-10000, ...) count as their base method."""
    return m.split("-")[0] in SEARCH_METHODS


class Construction(TypedDict):
    """One dpds row: its location, parameters, group, and connection set."""

    file: str
    line: int
    n: int
    k: int
    t: int
    lam: int
    mu: int
    lib: int
    members: str
    idx: list[int]
    method: str


# (n, lib, k, t, lambda, mu) identifying a (group, parameter) pair.
GroupKey = tuple[int, int, int, int, int, int]
# (key, method, searches num_dpds or None if no such search row, dpds count).
Problem = tuple[GroupKey, str, int | None, int]

# GAP driver for check 1. Per construction line "n lib k t lambda mu i1 i2 ...",
# tally f(g) = #{(s1,s2) in S^2 : s1*s2 = g} over the group's multiplication
# table (built once per group) and confirm the DSRG conditions: f(e)=t, f=lambda
# on S, f=mu elsewhere. The k^2 products and their tally are done with kernel ops
# (sublist + Collected), so per-construction interpreted work is only O(n). Prints
# one "RES\tOK"/"RES\tBAD" per construction; GAP flushes per newline for the bar.
CHECK = r"""
inf := InputTextFile("{inf}");; lines := [];;
l := ReadLine(inf);; while l <> fail do Add(lines, Chomp(l)); l := ReadLine(inf); od;;
CloseStream(inf);;
curkey := "";; M := 0;; n := 0;;
for line in lines do
  if Length(line) = 0 then continue; fi;
  p := SplitString(line, " ");;
  nn := Int(p[1]);; ll := Int(p[2]);; kk := Int(p[3]);;
  tt := Int(p[4]);; lam := Int(p[5]);; mu := Int(p[6]);;
  idx := List(p{{[7 .. Length(p)]}}, Int);;
  key := Concatenation(String(nn), "_", String(ll));;
  if key <> curkey then
    G := SmallGroup(nn, ll);; els := Elements(G);; n := nn;;
    M := List([1 .. n], i -> List([1 .. n], j -> Position(els, els[i] * els[j])));;
    curkey := key;;
  fi;
  inset := BlistList([1 .. n], idx);;
  f := ListWithIdenticalEntries(n, 0);;
  for pair in Collected(Concatenation(List(idx, i -> M[i]{{idx}}))) do
    f[pair[1]] := pair[2];;
  od;;
  ok := (Length(idx) = kk) and (not inset[1]) and (f[1] = tt);;
  pp := 2;;
  while ok and pp <= n do
    if inset[pp] then ok := f[pp] = lam; else ok := f[pp] = mu; fi;
    pp := pp + 1;;
  od;;
  if ok then Print("RES\tOK\n"); else Print("RES\tBAD\n"); fi;;
od;;
"""


def gap_stream(script: str, lines: list[str], desc: str, unit: str) -> list[str]:
    """Run a streaming GAP `script` over `lines`, advancing a tqdm bar as each
    "RES\\t<payload>" line arrives; return the payloads in order."""
    with tempfile.TemporaryDirectory() as d:
        inf, gf = Path(d) / "in", Path(d) / "s.g"
        inf.write_text("\n".join(lines) + "\n")
        gf.write_text(script.format(inf=inf))
        proc = subprocess.Popen(
            ["gap", "-q", str(gf)],
            stdin=subprocess.DEVNULL,  # else GAP enters its REPL on the TTY and never exits
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        out: list[str] = []
        assert proc.stdout is not None
        with tqdm(total=len(lines), desc=desc, unit=unit) as bar:
            for line in proc.stdout:
                if line.startswith("RES\t"):
                    out.append(line[4:].rstrip("\n"))
                    bar.update(1)
        if proc.wait():
            raise subprocess.CalledProcessError(proc.returncode, "gap")
    return out


def read_constructions(d: Path) -> list[Construction]:
    """Read every dpds row; sorted by (n, lib) so GAP reuses each group's table."""
    cons: list[Construction] = []
    for fn in sorted(glob.glob(str(d / "dpds" / "n*.csv"))):
        name = os.path.basename(fn)
        with open(fn, newline="") as f:
            for lineno, r in enumerate(csv.DictReader(f), start=2):  # row 1 is the header
                cons.append({
                    "file": name,
                    "line": lineno,
                    "n": int(r["n"]),
                    "k": int(r["k"]),
                    "t": int(r["t"]),
                    "lam": int(r["lambda"]),
                    "mu": int(r["mu"]),
                    "lib": int(r["lib_id"]),
                    "members": r["members"],
                    "idx": [int(x) for x in r["members"].split()],
                    "method": r["source_method"],
                })
    cons.sort(key=lambda c: (c["n"], c["lib"]))
    return cons


def check_ascending(cons: list[Construction]) -> list[Construction]:
    """Return constructions whose members are not strictly ascending."""
    return [c for c in cons if any(a >= b for a, b in zip(c["idx"], c["idx"][1:]))]


def check_num_dpds(d: Path, cons: list[Construction]) -> list[Problem]:
    """Return (key, method, searches_num_dpds, dpds_count) tuples that disagree.
    A searches value of None means constructions exist with no searches row."""
    permethod: dict[GroupKey, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for c in cons:
        key: GroupKey = (c["n"], c["lib"], c["k"], c["t"], c["lam"], c["mu"])
        for m in (x.strip() for x in c["method"].split(",")):
            permethod[key][m] += 1

    problems: list[Problem] = []
    seen: set[tuple[GroupKey, str]] = set()
    with open(d / "searches.csv", newline="") as f:
        for r in csv.DictReader(f):
            key = (
                int(r["n"]),
                int(r["lib_id"]),
                int(r["k"]),
                int(r["t"]),
                int(r["lambda"]),
                int(r["mu"]),
            )
            m, want = r["method"], int(r["num_dpds"])
            seen.add((key, m))
            got = permethod.get(key, {}).get(m, 0)
            if want != got:
                problems.append((key, m, want, got))
    for key, counts in permethod.items():
        for m, got in counts.items():
            if is_search_method(m) and (key, m) not in seen:
                problems.append((key, m, None, got))
    return problems


def check_is_dsrg(cons: list[Construction]) -> list[Construction]:
    """Return constructions that are not DSRGs with their stated parameters."""
    lines = [
        f"{c['n']} {c['lib']} {c['k']} {c['t']} {c['lam']} {c['mu']} {c['members']}" for c in cons
    ]
    status = gap_stream(CHECK, lines, "verifying DSRGs", "dpd")
    if len(status) != len(cons):
        raise RuntimeError(f"GAP returned {len(status)} results for {len(cons)} constructions")
    return [c for c, s in zip(cons, status) if s != "OK"]


def report[T](title: str, bad: list[T], fmt: Callable[[T], str], limit: int = 10) -> bool:
    """Print a check result; return True if it passed (no failures)."""
    if not bad:
        print(f"  OK   {title}")
        return True
    print(f"  FAIL {title}: {len(bad)} problem(s)")
    for item in bad[:limit]:
        print(f"         {fmt(item)}")
    if len(bad) > limit:
        print(f"         … and {len(bad) - limit} more")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a source-of-truth data dir.")
    ap.add_argument("data_dir", type=Path, help="directory holding searches.csv and dpds/")
    args = ap.parse_args()
    d: Path = args.data_dir.resolve()

    print(f"Verifying {d}", file=sys.stderr)
    cons = read_constructions(d)
    print(
        f"  {len(cons)} constructions across {len(glob.glob(str(d / 'dpds' / 'n*.csv')))} shards",
        file=sys.stderr,
    )

    ok = True
    print("Checking members are ascending …", file=sys.stderr)
    ok &= report(
        "members ascending",
        check_ascending(cons),
        lambda c: f"{c['file']}:{c['line']} members not ascending: {c['members']}",
    )
    print("Checking searches.num_dpds against dpds counts …", file=sys.stderr)
    ok &= report(
        "searches.num_dpds vs dpds counts",
        check_num_dpds(d, cons),
        lambda p: (
            f"group=SmallGroup({p[0][0]},{p[0][1]}) param={p[0][2:]} method={p[1]}: "
            f"searches={p[2]} dpds={p[3]}"
        ),
    )
    print("Checking every dpds is a DSRG (via GAP) …", file=sys.stderr)
    ok &= report(
        "every dpds is a DSRG",
        check_is_dsrg(cons),
        lambda c: (
            f"{c['file']}:{c['line']} SmallGroup({c['n']},{c['lib']}) "
            f"({c['n']},{c['k']},{c['t']},{c['lam']},{c['mu']}) members={c['members']}"
        ),
    )

    print("All checks passed." if ok else "Some checks FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
