#!/usr/bin/env python3
"""Command-line interface for the Cayley (D)SRG ILP model.

Two subcommands:

* ``solve``  — solve one instance for a single group SmallGroup(n, lib_id)
* ``search`` — solve one instance across every group of order n
* ``sweep``  — search across groups for every parameter set in a CSV

Examples::

    # Single group, DSRG:
    python -m ilp solve --n 48 --k 9 --t 5 --lambda 0 --mu 2 --lib-id 3

    # Single group, undirected SRG, with a time limit:
    python -m ilp solve --n 16 --k 6 --t 6 --lambda 2 --mu 2 --lib-id 3 \
        --undirected --time-limit 300

    # All non-abelian groups of order 48:
    python -m ilp search --n 48 --k 9 --t 5 --lambda 0 --mu 2

    # Every open case in a parameter table (order 48 only):
    python -m ilp sweep dsrg_params.csv --filter-n 48 --time-limit 300

    # Emit machine-readable JSON to a file:
    python -m ilp solve --n 48 --k 9 --t 5 --lambda 0 --mu 2 --lib-id 3 \
        --json --output result.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from ilp.cayley_dsrg import search_all_groups, solve_cayley_dsrg

# -- Shared options ---------------------------------------------------------------------


def _add_common_params(p: argparse.ArgumentParser) -> None:
    """Parameter-set arguments common to both subcommands."""
    p.add_argument("--n", type=int, required=True, help="Group order (= vertices)")
    p.add_argument("--k", type=int, required=True, help="In-/out-degree")
    p.add_argument("--t", type=int, required=True, help="Reciprocal neighbours per vertex")
    p.add_argument(
        "--lambda", dest="lambda_param", type=int, required=True, help="lambda parameter"
    )
    p.add_argument("--mu", type=int, required=True, help="mu parameter")


def _add_solver_opts(p: argparse.ArgumentParser) -> None:
    """Solver/formulation options common to both subcommands."""
    p.add_argument(
        "--undirected",
        action="store_true",
        help="SRG mode: enforce S = S^-1 (include abelian groups in search)",
    )
    p.add_argument(
        "--quadratic",
        action="store_true",
        help="Use the quadratic formulation instead of the linear one",
    )
    p.add_argument(
        "--no-aut-pruning", action="store_true", help="Disable Aut(G) lex-leader symmetry breaking"
    )
    p.add_argument("--threads", type=int, default=-1, help="Solver threads (-1 = Gurobi default)")
    p.add_argument(
        "--time-limit", type=float, default=None, help="Wall-clock limit per solve, in seconds"
    )
    p.add_argument(
        "--heuristics", type=float, default=None, help="Fraction of time spent on MIP heuristics"
    )
    p.add_argument(
        "-G",
        dest="gurobi_params",
        action="append",
        metavar="PARAM=VALUE",
        help="Extra Gurobi parameter override (repeatable)",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress Gurobi solver output")
    p.add_argument(
        "--json", action="store_true", help="Print result(s) as JSON instead of a summary table"
    )
    p.add_argument("--output", type=Path, default=None, help="Write JSON result(s) to this file")
    p.add_argument(
        "--dpds-out",
        type=Path,
        default=None,
        help=(
            "Write the found connection sets to a dpds-schema CSV "
            "(n,k,t,lambda,mu,lib_id,members,source_method); members 1-based per "
            "data/schema.md. One invocation is a single order n, so the file drops "
            "straight into data/dpds/nNNN.csv"
        ),
    )
    p.add_argument(
        "--searches-out",
        type=Path,
        default=None,
        help=(
            "Write one searches-schema row per swept group -- including the negatives -- "
            "to a CSV (n,k,t,lambda,mu,lib_id,method,outcome,num_dpds)"
        ),
    )


def _parse_gurobi_params(items: list[str] | None) -> dict[str, str]:
    """Turn a list of ``PARAM=VALUE`` strings into a dict."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"Invalid -G override (expected PARAM=VALUE): {item!r}")
        key, val = item.split("=", 1)
        out[key.strip()] = val.strip()
    return out


# -- Parameter CSV ----------------------------------------------------------------------


def _read_param_sets(
    csv_path: Path,
    filter_n: int | None,
    filter_status: str | None,
) -> list[dict]:
    """Read (and filter) parameter sets from a CSV or Excel file.

    Expected columns: ``n, k, lambda, mu`` and optionally ``t`` (defaults to
    ``k``) and ``Status``. Rows with ``k / n >= 0.5`` are dropped to avoid
    re-searching complementary graphs.
    """
    from typing import cast

    import pandas as pd

    reader = pd.read_excel if csv_path.suffix in (".xls", ".xlsx") else pd.read_csv
    df = cast("pd.DataFrame", reader(csv_path))
    df.columns = df.columns.str.strip()

    if filter_n is not None:
        df = df[df["n"] == filter_n]
    if filter_status is not None and "Status" in df.columns:
        df = df[df["Status"] == filter_status]
    df = cast("pd.DataFrame", df[df["k"] / df["n"] < 0.5])

    has_t = "t" in df.columns
    return [
        {
            "n": int(r["n"]),
            "k": int(r["k"]),
            "t": int(r["t"]) if has_t else int(r["k"]),
            "lambda": int(r["lambda"]),
            "mu": int(r["mu"]),
        }
        for _, r in df.iterrows()
    ]


# -- Output -----------------------------------------------------------------------------


def _print_summary(result: dict) -> None:
    """Human-readable one-instance summary."""
    kind = "srg" if result.get("undirected") else "dsrg"
    n, k, t = result["n"], result["k"], result["t"]
    lam, mu = result["lambda"], result["mu"]
    print(
        f"{result['status']:<14} "
        f"{kind}({n},{k},{t},{lam},{mu}) "
        f"g{result['lib_id']} ({result['group_name']})  "
        f"{result['wall_seconds']:.1f}s"
    )
    cs = result.get("connection_set_original_indices") or result.get("connection_set")
    if cs is not None:
        print(f"  connection set S = {cs}")


def _outcome(result: dict) -> str | None:
    """Map a solver status onto a searches-schema outcome.

    The model has a constant objective, so it asks a pure feasibility question:
    a solve that produced a connection set has answered it even if it stopped on
    the time limit, and counts as ``found`` rather than ``timeout``.

    Returns:
        A searches.outcome value, or None for a solve that recorded no search
        (an error, or an unrecognised status).
    """
    if result.get("connection_set_original_indices"):
        return "found"
    status = result.get("status")
    if status == "Infeasible":
        return "infeasible_proof"
    if status in {"TimeLimit", "Suboptimal"}:
        return "timeout"
    return None


def _write_csvs(results: list[dict], args: argparse.Namespace) -> None:
    """Write dpds/searches CSVs in the schema `rrhc` and data/ use.

    ``members`` is 1-based into ``Elements(G)`` and strictly ascending, matching
    what rrhc writes (``e + 1``) and what verify_data.py checks; the solver
    reports 0-based positions, hence the shift.
    """
    if args.searches_out is not None:
        with args.searches_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "k", "t", "lambda", "mu", "lib_id", "method", "outcome", "num_dpds"])
            for r in results:
                outcome = _outcome(r)
                if outcome is None:
                    print(
                        f"  skipping g{r['lib_id']}: no search recorded ({r['status']})",
                        file=sys.stderr,
                    )
                    continue
                w.writerow([
                    r["n"], r["k"], r["t"], r["lambda"], r["mu"], r["lib_id"],
                    "ilp", outcome, 1 if outcome == "found" else 0,
                ])
        print(f"Wrote {args.searches_out}", file=sys.stderr)

    if args.dpds_out is not None:
        with args.dpds_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "k", "t", "lambda", "mu", "lib_id", "members", "source_method"])
            for r in results:
                if _outcome(r) != "found":
                    continue
                members = sorted(i + 1 for i in r["connection_set_original_indices"])
                w.writerow([
                    r["n"], r["k"], r["t"], r["lambda"], r["mu"], r["lib_id"],
                    " ".join(str(i) for i in members), "ilp",
                ])
        print(f"Wrote {args.dpds_out}", file=sys.stderr)


def _emit(results: list[dict], args: argparse.Namespace) -> None:
    """Emit results as JSON and/or a summary, per CLI flags."""
    _write_csvs(results, args)

    if args.output is not None:
        payload = results[0] if len(results) == 1 else results
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.output}", file=sys.stderr)

    if args.json:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))
    else:
        for r in results:
            _print_summary(r)


# -- Subcommands ------------------------------------------------------------------------


def cmd_solve(args: argparse.Namespace) -> None:
    result = solve_cayley_dsrg(
        args.n,
        args.k,
        args.t,
        args.lambda_param,
        args.mu,
        args.lib_id,
        use_aut_pruning=not args.no_aut_pruning,
        undirected=args.undirected,
        quadratic=args.quadratic,
        threads=args.threads,
        time_limit=args.time_limit,
        heuristics=args.heuristics,
        quiet=args.quiet,
        gurobi_params=_parse_gurobi_params(args.gurobi_params),
    )
    result.setdefault("undirected", args.undirected)
    _emit([result], args)


def cmd_search(args: argparse.Namespace) -> None:
    results = search_all_groups(
        args.n,
        args.k,
        args.t,
        args.lambda_param,
        args.mu,
        use_aut_pruning=not args.no_aut_pruning,
        undirected=args.undirected,
        threads=args.threads,
        time_limit=args.time_limit,
        quiet=args.quiet,
        gurobi_params=_parse_gurobi_params(args.gurobi_params),
        include_abelian=args.undirected,
    )
    for r in results:
        r.setdefault("undirected", args.undirected)
    _emit(results, args)


def cmd_sweep(args: argparse.Namespace) -> None:
    param_sets = _read_param_sets(args.params_csv, args.filter_n, args.filter_status)
    if not param_sets:
        raise SystemExit("No parameter sets match the given filters.")

    gurobi_params = _parse_gurobi_params(args.gurobi_params)
    results: list[dict] = []
    for i, p in enumerate(param_sets, 1):
        if not args.json:
            print(
                f"# [{i}/{len(param_sets)}] "
                f"n={p['n']} k={p['k']} t={p['t']} lambda={p['lambda']} mu={p['mu']}",
                file=sys.stderr,
            )
        group_results = search_all_groups(
            p["n"],
            p["k"],
            p["t"],
            p["lambda"],
            p["mu"],
            use_aut_pruning=not args.no_aut_pruning,
            undirected=args.undirected,
            threads=args.threads,
            time_limit=args.time_limit,
            quiet=args.quiet,
            gurobi_params=gurobi_params,
            include_abelian=args.undirected,
        )
        for r in group_results:
            r.setdefault("undirected", args.undirected)
        results.extend(group_results)
    _emit(results, args)


# -- Entry point ------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ilp",
        description="ILP search for Cayley-graph (directed) strongly regular graphs.",
    )
    sub = parser.add_subparsers(dest="command", title="subcommands", required=True)

    p_solve = sub.add_parser("solve", help="Solve one instance for a single group")
    _add_common_params(p_solve)
    p_solve.add_argument("--lib-id", type=int, required=True, help="GAP SmallGroup library ID")
    _add_solver_opts(p_solve)
    p_solve.set_defaults(func=cmd_solve)

    p_search = sub.add_parser("search", help="Solve one instance across all groups of order n")
    _add_common_params(p_search)
    _add_solver_opts(p_search)
    p_search.set_defaults(func=cmd_search)

    p_sweep = sub.add_parser("sweep", help="Search across groups for every parameter set in a CSV")
    p_sweep.add_argument("params_csv", type=Path, help="CSV/Excel with n,k,[t,]lambda,mu columns")
    p_sweep.add_argument("--filter-n", type=int, default=None, help="Only rows with this n")
    p_sweep.add_argument(
        "--filter-status", default=None, help="Only rows with this Status (e.g. 'open')"
    )
    _add_solver_opts(p_sweep)
    p_sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
