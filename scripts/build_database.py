"""Build dpds.sqlite from a source-of-truth data dir.

The data dir must hold `parameters.csv`, `searches.csv`, and a `dpds/` directory
of `nNNN.csv` shards (see data2/ for an example). The database follows
data/schema.sql. The derived tables (groups, group_param, graphs) and every
`digraph6` are recomputed from scratch via GAP rather than read from a CSV.

If `<data_dir>/logs/` exists (see scripts/extract_hpc_logs.py), its run-log
extracts are loaded into the ilp_solves, rrhc_best_error and rrhc_task_times tables.

Usage: build_database.py <data_dir>   ->   writes <data_dir>/dpds.sqlite
"""

import argparse
import csv
import glob
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from typing import TypedDict

from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
SCHEMA = REPO / "data" / "schema.sql"
# Outcomes that definitively resolve a (group, parameter) pair.
PROOF_OUTCOMES: set[str] = {"found", "infeasible_proof", "empty_proof"}


class Con(TypedDict):
    """One dpds construction: group, parameter (as param_id), and connection set."""

    n: int
    lib: int
    pid: int
    members: str
    method: str


Key = tuple[str, str, str, str, str]  # (n, k, t, lambda, mu) as CSV strings
Group = tuple[int, int]  # (n, lib_id)
SearchRec = tuple[int, int, int, str, str, int]  # (n, lib, param_id, method, outcome, num_dpds)

# Row tuples, in the column order of the corresponding table in data/schema.sql.
ParamRow = tuple[int, int, int, int, int, int, str]
GroupRow = tuple[int, int, int, str, int]
GpRow = tuple[int, int]
GraphRow = tuple[int, str, int, int, str, int, int, int, int]
DpdsRow = tuple[int, int, str, str, str, int]
SearchRow = tuple[int, int, str, str, int, int, int]
LogTables = dict[str, list[tuple]]

# --- GAP drivers ------------------------------------------------------------
# Templated with {inf} (input path); literal braces are doubled for str.format.
# Streaming drivers print one "RES\t<payload>" line per item to stdout as it is
# computed (GAP flushes per newline), which drives a progress bar; anything else
# on stdout (package-load banner, etc.) is ignored by the runner.

# Canonical digraph6 per construction line "n lib_id i1 i2 ...". The group's
# multiplication table M[i][j] = position of els[i]*els[j] is built once per
# group and reused, so each construction is O(n^2) table lookups.
CANON = r"""
LoadPackage("digraphs");;
SetPrintFormattingStatus("*stdout*", false);;  # no line-wrapping: keep each RES on one line
inf := InputTextFile("{inf}");; lines := [];;
l := ReadLine(inf);; while l <> fail do Add(lines, Chomp(l)); l := ReadLine(inf); od;;
CloseStream(inf);;
curkey := "";; M := 0;; n := 0;;
for line in lines do
  if Length(line) = 0 then continue; fi;
  p := SplitString(line, " ");;
  nn := Int(p[1]);; ll := Int(p[2]);; key := Concatenation(String(nn), "_", String(ll));;
  if key <> curkey then
    G := SmallGroup(nn, ll);; els := Elements(G);; n := nn;;
    M := List([1 .. n], i -> List([1 .. n], j -> Position(els, els[i] * els[j])));;
    curkey := key;;
  fi;
  idx := List(p{{[3 .. Length(p)]}}, Int);;
  D := Digraph(List([1 .. n], i -> M[i]{{idx}}));;
  Print("RES\t", Digraph6String(BlissCanonicalDigraph(D)), "\n");;
od;;
"""

# Per distinct digraph6: aut_order, is_drr (aut == n), is_self_converse.
INV = r"""
LoadPackage("digraphs");;
SetPrintFormattingStatus("*stdout*", false);;  # no line-wrapping: keep each RES on one line
inf := InputTextFile("{inf}");;
l := ReadLine(inf);;
while l <> fail do
  s := Chomp(l);;
  if Length(s) > 0 then
    D := DigraphFromDigraph6String(s);;
    a := Order(AutomorphismGroup(D));;
    sc := IsIsomorphicDigraph(D, DigraphReverse(D));;
    drr := a = DigraphNrVertices(D);;
    Print("RES\t", s, "\t", String(a), "\t", String(drr), "\t", String(sc), "\n");;
  fi;
  l := ReadLine(inf);;
od;;
"""

# Per order n (one per input line), one result per group:
# "RES\tn\tlib_id\tis_abelian\tStructureDescription" for every SmallGroup(n, *).
# StructureDescription is slow, so streaming a result per group drives the bar.
META = r"""
SetPrintFormattingStatus("*stdout*", false);;  # no line-wrapping: keep each RES on one line
inf := InputTextFile("{inf}");;
l := ReadLine(inf);;
while l <> fail do
  s := Chomp(l);;
  if Length(s) > 0 then
    n := Int(s);;
    for i in [1 .. NrSmallGroups(n)] do
      G := SmallGroup(n, i);;
      Print("RES\t", String(n), "\t", String(i), "\t",
            String(IsAbelian(G)), "\t", StructureDescription(G), "\n");;
    od;
  fi;
  l := ReadLine(inf);;
od;;
"""


def gap_stream(script: str, lines: list[str], desc: str, unit: str, total: int | None) -> list[str]:
    """Run a streaming GAP `script` over `lines`, advancing a tqdm bar as each
    "RES\\t<payload>" line arrives; return the payloads in order. `total` is the
    expected number of results (None for an indeterminate bar when the script
    emits a variable number of results per input line)."""
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
        with tqdm(total=total, desc=desc, unit=unit) as bar:
            for line in proc.stdout:
                if line.startswith("RES\t"):  # a result, not GAP banner noise
                    out.append(line[4:].rstrip("\n"))
                    bar.update(1)
        if proc.wait():
            raise subprocess.CalledProcessError(proc.returncode, "gap")
    return out


def read_parameters(d: Path) -> tuple[list[ParamRow], dict[Key, int]]:
    """Return (rows, key2pid): rows match `parameters`, key2pid maps the
    (n,k,t,lambda,mu) string tuple -> param_id."""
    rows: list[ParamRow] = []
    key2pid: dict[Key, int] = {}
    with open(d / "parameters.csv", newline="") as f:
        for pid, r in enumerate(csv.DictReader(f), 1):
            key: Key = (r["n"], r["k"], r["t"], r["lambda"], r["mu"])
            key2pid[key] = pid
            rows.append((
                pid,
                int(r["n"]),
                int(r["k"]),
                int(r["t"]),
                int(r["lambda"]),
                int(r["mu"]),
                r["status"],
            ))
    return rows, key2pid


def read_searches(
    d: Path, key2pid: dict[Key, int]
) -> tuple[list[SearchRec], set[Group], set[int]]:
    """Return (rows, referenced_groups, orders) from searches.csv."""
    rows: list[SearchRec] = []
    refs: set[Group] = set()
    ns: set[int] = set()
    with open(d / "searches.csv", newline="") as f:
        for r in csv.DictReader(f):
            n, lib = int(r["n"]), int(r["lib_id"])
            pid = key2pid[(r["n"], r["k"], r["t"], r["lambda"], r["mu"])]
            rows.append((n, lib, pid, r["method"], r["outcome"], int(r["num_dpds"])))
            refs.add((n, lib))
            ns.add(n)
    return rows, refs, ns


def read_dpds(d: Path, key2pid: dict[Key, int]) -> tuple[list[Con], set[Group]]:
    """Return (constructions, referenced_groups); constructions sorted by group."""
    cons: list[Con] = []
    refs: set[Group] = set()
    for fn in sorted(glob.glob(str(d / "dpds" / "n*.csv"))):
        with open(fn, newline="") as f:
            for r in csv.DictReader(f):
                n, lib = int(r["n"]), int(r["lib_id"])
                cons.append({
                    "n": n,
                    "lib": lib,
                    "pid": key2pid[(r["n"], r["k"], r["t"], r["lambda"], r["mu"])],
                    "members": r["members"],
                    "method": r["source_method"],
                })
                refs.add((n, lib))
    cons.sort(key=lambda c: (c["n"], c["lib"]))
    return cons, refs


def group_metadata(ns: set[int]) -> tuple[dict[Group, tuple[int, str]], set[Group]]:
    """Return (meta, nonabelian) for every group of each order in `ns`.
    meta: (n,lib_id) -> (is_abelian, name); nonabelian: set of (n,lib_id)."""
    meta: dict[Group, tuple[int, str]] = {}
    nonab: set[Group] = set()
    for line in gap_stream(META, [str(n) for n in sorted(ns)], "group metadata", "group", None):
        n, lib, ab, name = line.split("\t")
        g: Group = (int(n), int(lib))
        is_ab = int(ab == "true")
        meta[g] = (is_ab, name)
        if not is_ab:
            nonab.add(g)
    return meta, nonab


def assemble_graphs(
    cons: list[Con], d6s: list[str]
) -> tuple[
    list[str],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, set[Group]],
]:
    """Deduplicate constructions by digraph6 into graph classes.
    Return (order, gid, pid, n_of, num_dpds, groups) keyed by digraph6."""
    order: list[str] = []
    gid: dict[str, int] = {}
    pid: dict[str, int] = {}
    n_of: dict[str, int] = {}
    ndpds: dict[str, int] = {}
    groups: dict[str, set[Group]] = {}
    for c, d6 in zip(cons, d6s):
        if d6 not in gid:
            gid[d6] = len(order) + 1
            order.append(d6)
            pid[d6] = c["pid"]
            n_of[d6] = c["n"]
            ndpds[d6] = 0
            groups[d6] = set()
        ndpds[d6] += 1
        groups[d6].add((c["n"], c["lib"]))
    return order, gid, pid, n_of, ndpds, groups


def read_logs(d: Path, key2pid: dict[Key, int], gid_of: dict[Group, int]) -> LogTables:
    """Rows for the run-log tables from <d>/logs/*.csv (empty if absent)."""
    out: LogTables = {"ilp_solves": [], "rrhc_best_error": [], "rrhc_task_times": []}
    logs = d / "logs"

    def rows(name: str) -> list[dict[str, str]]:
        f = logs / f"{name}.csv"
        if not f.exists():
            return []
        with open(f, newline="") as fh:
            return list(csv.DictReader(fh))

    def pid(r: dict[str, str]) -> int:
        return key2pid[(r["n"], r["k"], r["t"], r["lambda"], r["mu"])]

    def gid(r: dict[str, str]) -> int:
        return gid_of[(int(r["n"]), int(r["lib_id"]))]

    for r in rows("ilp_solves"):
        out["ilp_solves"].append((gid(r), pid(r), r["method"], r["status"], float(r["seconds"])))
    for r in rows("rrhc_best_error"):
        out["rrhc_best_error"].append((gid(r), pid(r), r["method"], int(r["best_error"])))
    for r in rows("rrhc_task_times"):
        out["rrhc_task_times"].append(
            (pid(r), r["method"], int(r["num_groups"]), int(r["elapsed_s"]), int(r["ncpus"]))
        )
    return out


def build_db(
    dbpath: Path,
    params: list[ParamRow],
    groups_rows: list[GroupRow],
    gp_rows: list[GpRow],
    graphs_rows: list[GraphRow],
    dpds_rows: list[DpdsRow],
    search_rows: list[SearchRow],
    log_rows: LogTables,
) -> None:
    if dbpath.exists():
        dbpath.unlink()
    con = sqlite3.connect(dbpath)
    con.executescript(SCHEMA.read_text())
    con.executemany("INSERT INTO parameters VALUES (?,?,?,?,?,?,?)", params)
    con.executemany("INSERT INTO groups VALUES (?,?,?,?,?)", groups_rows)
    con.executemany("INSERT INTO group_param VALUES (?,?)", gp_rows)
    con.executemany("INSERT INTO graphs VALUES (?,?,?,?,?,?,?,?,?)", graphs_rows)
    con.executemany(
        "INSERT INTO dpds(group_id,param_id,members,digraph6,source_method,graph_id) "
        "VALUES (?,?,?,?,?,?)",
        dpds_rows,
    )
    con.executemany(
        "INSERT INTO searches(group_id,param_id,method,outcome,is_proof,num_dpds,num_records) "
        "VALUES (?,?,?,?,?,?,?)",
        search_rows,
    )
    con.executemany("INSERT INTO ilp_solves VALUES (?,?,?,?,?)", log_rows["ilp_solves"])
    con.executemany("INSERT INTO rrhc_best_error VALUES (?,?,?,?)", log_rows["rrhc_best_error"])
    con.executemany("INSERT INTO rrhc_task_times VALUES (?,?,?,?,?)", log_rows["rrhc_task_times"])
    con.commit()
    con.close()


def log(msg: str) -> None:
    """Print a phase message to stderr (kept off stdout, alongside the bars)."""
    print(msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build dpds.sqlite from a source-of-truth data dir.")
    ap.add_argument(
        "data_dir", type=Path, help="directory holding parameters.csv, searches.csv, and dpds/"
    )
    ap.add_argument(
        "-o", "--output", type=Path, help="output database path (default: <data_dir>/dpds.sqlite)"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    d: Path = args.data_dir.resolve()
    dbpath: Path = (args.output or d / "dpds.sqlite").resolve()

    log(f"[1/5] Reading source CSVs from {d} …")
    params, key2pid = read_parameters(d)
    searches, sref, ns = read_searches(d, key2pid)
    cons, dref = read_dpds(d, key2pid)
    log(f"      {len(params)} parameters, {len(searches)} searches, {len(cons)} constructions")

    # Groups: those referenced by searches/dpds plus every nonabelian group of a
    # parameter order (the candidate grid). Assign group_id in sorted order.
    orders = ns | {p[1] for p in params}
    log(f"[2/5] Recomputing group metadata for {len(orders)} orders via GAP …")
    meta, nonab = group_metadata(orders)
    log(f"      {len(meta)} groups ({len(nonab)} nonabelian)")
    needed = sorted(sref | dref | nonab)
    gid_of = {g: i for i, g in enumerate(needed, 1)}
    groups_rows = [(gid_of[g], g[0], g[1], meta[g][1], meta[g][0]) for g in needed]

    # group_param: every nonabelian group of order n paired with every param of n.
    nonab_by_n: dict[int, list[int]] = {}
    for n, lib in nonab:
        nonab_by_n.setdefault(n, []).append(lib)
    gp_rows = [(gid_of[(p[1], lib)], p[0]) for p in params for lib in nonab_by_n.get(p[1], [])]

    # Graphs: recompute canonical digraph6 per construction, dedup, add invariants.
    log("[3/5] Recomputing canonical digraph6 for every construction …")
    d6s = gap_stream(
        CANON, [f"{c['n']} {c['lib']} {c['members']}" for c in cons], "canonicalizing", "dpd",
        len(cons),
    )
    order, gid, pid, n_of, ndpds, grps = assemble_graphs(cons, d6s)
    log(f"[4/5] Computing invariants for {len(order)} distinct graphs …")
    inv: dict[str, tuple[str, int, int]] = {}
    for line in gap_stream(INV, order, "invariants", "graph", len(order)):
        s, a, drr, sc = line.split("\t")
        inv[s] = (a, int(drr == "true"), int(sc == "true"))
    graphs_rows = [(gid[s], s, pid[s], n_of[s], *inv[s], ndpds[s], len(grps[s])) for s in order]

    dpds_rows = [
        (gid_of[(c["n"], c["lib"])], c["pid"], c["members"], d6, c["method"], gid[d6])
        for c, d6 in zip(cons, d6s)
    ]
    # num_records is not carried in the source CSVs; use the found-count as a proxy.
    search_rows = [
        (gid_of[(n, lib)], pid, m, o, int(o in PROOF_OUTCOMES), nd, nd)
        for (n, lib, pid, m, o, nd) in searches
    ]

    log_rows = read_logs(d, key2pid, gid_of)
    log("      run logs: " + ", ".join(f"{k} {len(v)}" for k, v in log_rows.items()))

    log(f"[5/5] Writing {dbpath} …")
    build_db(dbpath, params, groups_rows, gp_rows, graphs_rows, dpds_rows, search_rows, log_rows)
    print(
        f"Wrote {dbpath}: {len(params)} params, {len(groups_rows)} groups, "
        f"{len(order)} graphs, {len(cons)} dpds, {len(searches)} searches"
    )


if __name__ == "__main__":
    main()
