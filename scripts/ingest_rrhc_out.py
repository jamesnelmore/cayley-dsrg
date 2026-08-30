"""Scan a search-output dir for searches/dpds rows not yet in the data dir.

Reads every `searches*.csv` and `dpds*.csv` under <data_dir> (e.g. the rrhc-out/
written by scripts/slurm/rrhc.sh), drops the rows already recorded in <merge_dir>,
and reports -- or, with --apply, appends -- what is new. Both file kinds must be
in the current data/schema.md layout:

    searches  n,k,t,lambda,mu,lib_id,method,outcome,num_dpds
    dpds      n,k,t,lambda,mu,lib_id,members,source_method

A searches row is identified by (n,k,t,lambda,mu,lib_id,method) -- one attempt
per group/parameter/method -- and a dpds row by (n,k,t,lambda,mu,lib_id,members),
with members sorted so the same connection set keys the same however it was found.

Usage: ingest_rrhc_out.py <data_dir> [merge_dir] [--apply]
"""

import argparse
import csv
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent

PARAMS = ["n", "k", "t", "lambda", "mu"]
SEARCH_COLS = PARAMS + ["lib_id", "method", "outcome", "num_dpds"]
DPDS_COLS = PARAMS + ["lib_id", "members", "source_method"]
SEARCH_KEY = PARAMS + ["lib_id", "method"]
DPDS_KEY = PARAMS + ["lib_id", "members"]


def canon_members(members: str) -> str:
    """Sort a connection set numerically -- data/dpds is uniformly sorted, so an
    unsorted set from a search would otherwise not match its stored twin."""
    return " ".join(str(i) for i in sorted(int(x) for x in members.split()))


def read_rows(path: Path, cols: list[str]) -> list[dict[str, str]]:
    """Read a CSV, projecting to `cols` and rejecting anything not in schema."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in cols if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{path}: not in the current schema (missing columns: {', '.join(missing)})")
        return [{c: r[c] for c in cols} for r in reader]


def load_existing(data_dir: Path) -> tuple[set[tuple], set[tuple]]:
    """Keys already recorded in data/: (searches, dpds)."""
    searches = {
        tuple(r[c] for c in SEARCH_KEY) for r in read_rows(data_dir / "searches.csv", SEARCH_COLS)
    }
    dpds = set()
    for shard in sorted((data_dir / "dpds").glob("n*.csv")):
        for r in read_rows(shard, DPDS_COLS):
            r["members"] = canon_members(r["members"])
            dpds.add(tuple(r[c] for c in DPDS_KEY))
    return searches, dpds


def new_only(rows: list[dict], key_cols: list[str], seen: set[tuple]) -> list[dict]:
    """Rows whose key is in neither `seen` nor an earlier row (mutates `seen`)."""
    out = []
    for row in rows:
        key = tuple(row[c] for c in key_cols)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def append(path: Path, cols: list[str], rows: list[dict]) -> None:
    """Append rows, writing the header first if the file is new or empty."""
    new_file = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("data_dir", type=Path, help="search-output directory to scan.")
    p.add_argument(
        "merge_dir",
        type=Path,
        nargs="?",
        default=REPO / "data",
        help="Directory to merge into (default: the repo's data/)",
    )
    p.add_argument(
        "--apply", action="store_true", help="write the new rows. Dry run is default behavior"
    )
    p.add_argument(
        "--method",
        help=(
            "Override every scanned row's method/source_method column with this "
            "value before keying/merging, e.g. rrhc-10000 or ilp-30. For output "
            "written before those columns encoded their budget (a bare rrhc/ilp) "
            "-- current rrhc/ilp writers already label their own rows and need no "
            "override. Applied to both searches.method and dpds.source_method: a "
            "bare tag on dpds can't tell one run's construction from another's, "
            "which is exactly what let one ingest silently erase another's."
        ),
    )
    args = p.parse_args()

    have_searches, have_dpds = load_existing(args.merge_dir)

    search_rows = [
        r for f in sorted(args.data_dir.glob("searches*.csv")) for r in read_rows(f, SEARCH_COLS)
    ]
    if args.method:
        for r in search_rows:
            r["method"] = args.method
    dpds_rows = []
    for f in sorted(args.data_dir.glob("dpds*.csv")):
        for row in read_rows(f, DPDS_COLS):
            row["members"] = canon_members(row["members"])
            if args.method:
                row["source_method"] = args.method
            dpds_rows.append(row)

    new_searches = new_only(search_rows, SEARCH_KEY, have_searches)
    new_dpds = new_only(dpds_rows, DPDS_KEY, have_dpds)

    print(f"searches: {len(new_searches)} new of {len(search_rows)} scanned")
    print(f"dpds:     {len(new_dpds)} new of {len(dpds_rows)} scanned")

    if not args.apply:
        print("\n(dry run -- rerun with --apply to write)")
        return

    if new_searches:
        append(args.merge_dir / "searches.csv", SEARCH_COLS, new_searches)
    # dpds is sharded by vertex count, so split the new rows by n.
    shards: dict[int, list[dict]] = {}
    for row in new_dpds:
        shards.setdefault(int(row["n"]), []).append(row)
    for n, rows in sorted(shards.items()):
        append(args.merge_dir / "dpds" / f"n{n:03d}.csv", DPDS_COLS, rows)
        print(f"  dpds/n{n:03d}.csv += {len(rows)}")
    print(f"\nwrote {len(new_searches)} searches, {len(new_dpds)} dpds rows to {args.merge_dir}")


if __name__ == "__main__":
    main()
