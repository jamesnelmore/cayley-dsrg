"""Cayley-graph-specific DSRG ILP formulation using Gurobi.

Instead of O(n^2) edge variables (general DSRG ILP), this exploits the Cayley
graph structure: a Cayley graph Cay(G, S) has edge g->h iff g^{-1}h in S, so
the adjacency matrix is fully determined by the connection set S.  We only
need n-1 binary variables x_g indicating whether each non-identity group
element g is in S.

The DSRG condition at the identity (vertex-transitive, so sufficient to check
at a single vertex) becomes:

    For each h in G\\{e}:
        sum_{g in G\\{e}} x_g * x_{g^{-1}h} = (lambda - mu) * x_h + mu

Symmetry breaking uses Aut(G) — automorphisms permute elements and yield
isomorphic Cayley graphs, so we add lex-leader constraints.

Requires GAP (found via PATH or GAP_BIN env var) for group algebra computations.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

# -- GAP integration --------------------------------------------------------------------

GAP_SCRIPT = Path(__file__).with_name("cayley_dsrg.g")
GAP_BIN = shutil.which("gap") or "/opt/homebrew/bin/gap"


@dataclass
class CayleyGroupData:
    """Parsed output from the GAP script for a single group."""

    n: int
    lib_id: int
    name: str
    identity_pos: int
    num_nonid: int
    nonid_order: list[int]  # original 0-indexed positions
    inv_map: list[int]  # inv_map[i] = non-id index of inverse of element i
    involutions: list[int]  # non-id indices that are involutions
    pairs: list[tuple[int, int]]  # (i, j) with i < j, mutual inverses
    products: list[
        list[int]
    ]  # products[g][h] = non-id index of g^{-1}h, or -1 if identity
    aut_generators: list[list[int]]  # each generator is a permutation of non-id indices


def load_cayley_data(n: int, lib_id: int) -> CayleyGroupData:
    """Run GAP to compute group data for SmallGroup(n, lib_id)."""
    gap_input = f"n := {n};;\nlib_id := {lib_id};;\n" + GAP_SCRIPT.read_text()

    proc = subprocess.run(
        [GAP_BIN, "-q"],
        input=gap_input,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GAP failed (exit {proc.returncode}):\n{proc.stderr}")

    return _parse_gap_output(proc.stdout)


def _parse_gap_output(output: str) -> CayleyGroupData:
    """Parse the GAP script output into a CayleyGroupData.

    Handles GAP's automatic line wrapping for large groups by collecting
    all tokens between keyword markers.
    """
    lines = output.strip().splitlines()
    i = 0

    def _next() -> str:
        """Return the next non-empty line."""
        nonlocal i
        while i < len(lines) and not lines[i].strip():
            i += 1
        line = lines[i].strip()
        i += 1
        return line

    def _collect_until(marker: str) -> list[str]:
        """Collect all tokens from lines until a line starting with *marker*."""
        nonlocal i
        tokens: list[str] = []
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith(marker):
                return tokens
            tokens.extend(stripped.split())
            i += 1
        raise ValueError(f"Expected marker {marker!r} but hit end of output")

    def _collect_tagged_line(tag: str, next_marker: str) -> list[int]:
        """Collect integer tokens for a tagged line that may wrap.

        Reads the first line (which starts with *tag*), strips the tag,
        then continues collecting continuation lines until the next keyword.
        """
        nonlocal i
        line = _next()
        assert line.startswith(tag), f"Expected {tag}, got: {line}"
        # Tokens after the tag on the first line.
        first_tokens = line.split()[1:]
        # Continuation lines until next marker.
        cont_tokens = _collect_until(next_marker)
        return [int(x) for x in first_tokens + cont_tokens]

    # -- Parse header -------------------------------------------------------------------
    line = _next()
    assert line.startswith("CAYLEY_START"), f"Expected CAYLEY_START, got: {line}"
    parts = line.split(maxsplit=3)
    n = int(parts[1])
    lib_id = int(parts[2])
    name = parts[3] if len(parts) > 3 else "?"

    line = _next()
    identity_pos = int(line.split()[1])

    line = _next()
    num_nonid = int(line.split()[1])

    # Each of the following lines may wrap across multiple output lines.
    # We collect tokens until the next known keyword.
    nonid_order = _collect_tagged_line("NONID_ORDER", "INV_MAP")
    inv_map = _collect_tagged_line("INV_MAP", "INVOLUTIONS")
    involutions = _collect_tagged_line("INVOLUTIONS", "PAIRS")
    pair_tokens = _collect_tagged_line("PAIRS", "PRODUCTS_START")
    pairs = [
        (pair_tokens[j], pair_tokens[j + 1]) for j in range(0, len(pair_tokens), 2)
    ]

    # -- Products table -----------------------------------------------------------------
    line = _next()
    assert line == "PRODUCTS_START", f"Expected PRODUCTS_START, got: {line}"

    # Collect all tokens between PRODUCTS_START and PRODUCTS_END.
    all_prod_tokens = _collect_until("PRODUCTS_END")
    prod_ints = [int(x) for x in all_prod_tokens]
    products: list[list[int]] = []
    for row_idx in range(num_nonid):
        start = row_idx * num_nonid
        products.append(prod_ints[start : start + num_nonid])

    line = _next()
    assert line == "PRODUCTS_END", f"Expected PRODUCTS_END, got: {line}"

    # -- Automorphism generators --------------------------------------------------------
    line = _next()
    num_gens = int(line.split()[1])

    # Collect all aut generator tokens until CAYLEY_END.
    all_aut_tokens = _collect_until("CAYLEY_END")
    aut_ints = [int(x) for x in all_aut_tokens]
    aut_generators: list[list[int]] = []
    for gen_idx in range(num_gens):
        start = gen_idx * num_nonid
        aut_generators.append(aut_ints[start : start + num_nonid])

    line = _next()
    assert line == "CAYLEY_END", f"Expected CAYLEY_END, got: {line}"

    return CayleyGroupData(
        n=n,
        lib_id=lib_id,
        name=name,
        identity_pos=identity_pos,
        num_nonid=num_nonid,
        nonid_order=nonid_order,
        inv_map=inv_map,
        involutions=involutions,
        pairs=pairs,
        products=products,
        aut_generators=aut_generators,
    )


# -- Helpers ----------------------------------------------------------------------------


def _extract_status(model: gp.Model) -> str:
    return {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.SUBOPTIMAL: "Suboptimal",
    }.get(model.Status, f"Unknown({model.Status})")


# -- Model builder ----------------------------------------------------------------------


def build_cayley_dsrg(
    n: int,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
    group_data: CayleyGroupData,
    *,
    use_aut_pruning: bool = True,
    undirected: bool = False,
    quiet: bool = True,
) -> tuple[gp.Model, list[gp.Var]]:
    """Build a Cayley-graph DSRG ILP for a specific group.

    Variables:
        x[g] in {0,1} for each non-identity element g — whether g in S.

    Constraints:
        1. Degree: sum x[g] = k
        2. Reciprocal count: sum of reciprocal contributions = t
        3. DSRG 2-path condition at identity (one row suffices by
           vertex-transitivity):
           For each h in G\\{e}:
             sum_{g} x[g] * x[g^{-1}h] = (lambda-mu)*x[h] + mu

    Linearization uses deduplicated auxiliary variables for products.

    Args:
        n: Group order (= number of vertices).
        k: In- and out-degree.
        t: Reciprocal neighbours per vertex.
        lambda_param: 2-path count for adjacent pairs.
        mu: 2-path count for non-adjacent pairs.
        group_data: Pre-computed group algebra data from GAP.
        use_aut_pruning: Add Aut(G) lex-leader symmetry-breaking constraints.
        undirected: If True, enforce S = S^{-1} (for undirected Cayley / SRG search).
            Adds x[g] == x[g^{-1}] for all non-involution pairs; Gurobi presolve
            eliminates the redundant variables automatically.
        quiet: Suppress Gurobi output.

    Returns:
        (model, x_vars) where x_vars[i] is the binary variable for
        non-identity element i.
    """
    m = group_data.num_nonid  # n - 1
    inv_map = group_data.inv_map
    products = group_data.products
    lam_minus_mu = lambda_param - mu

    model = gp.Model(f"CayleyDSRG_{n}_{k}_{t}_{lambda_param}_{mu}_g{group_data.lib_id}")
    if quiet:
        model.setParam("OutputFlag", 0)

    # Feasibility problem.
    model.setObjective(0, GRB.MINIMIZE)

    # ── Decision variables: x[g] for g in G\{e} ─────────────────────────
    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{g}") for g in range(m)]

    # ── Constraint 1: Degree — sum x[g] = k ─────────────────────────────
    model.addConstr(
        gp.quicksum(x[g] for g in range(m)) == k,
        name="degree",
    )

    # ── Constraint 1b (undirected): S = S^{-1} ─────────────────────────
    if undirected:
        for g_lo, g_hi in group_data.pairs:
            model.addConstr(x[g_lo] == x[g_hi], name=f"sym_{g_lo}_{g_hi}")

    # ── Constraint 2: Reciprocal count — |S ∩ S^{-1}| = t ──────────────
    # t = |S ∩ S^{-1}| counts elements. For involutions (g = g^{-1}), x[g]
    # contributes 1. For non-involution pairs (g, g^{-1}), if both are in S
    # then BOTH elements are in S ∩ S^{-1}, contributing 2.
    #
    # sum_involutions x[g] + 2 * sum_pairs x[g]*x[g^{-1}] = t
    recip_expr = gp.LinExpr()

    for g in group_data.involutions:
        recip_expr += x[g]

    pair_aux: dict[tuple[int, int], gp.Var] = {}
    for g_lo, g_hi in group_data.pairs:
        r = model.addVar(vtype=GRB.BINARY, name=f"rpair_{g_lo}_{g_hi}")
        model.addGenConstrAnd(r, [x[g_lo], x[g_hi]], name=f"rpair_and_{g_lo}_{g_hi}")
        pair_aux[g_lo, g_hi] = r
        recip_expr += 2 * r

    model.addConstr(recip_expr == t, name="reciprocal")

    # ── Constraint 3: DSRG 2-path condition at identity ──────────────────
    # For each h in G\{e}:
    #   sum_{g in G\{e}} x[g] * x[products[g][h]] = lam_minus_mu * x[h] + mu
    #
    # where products[g][h] is the non-id index of g^{-1}h (or -1 if identity).
    #
    # If g^{-1}h = e, then x[products[g][h]] would be x[e] which is not in our
    # variable set. But g^{-1}h = e means h = g, so the term is x[g] * 1 = x[g]
    # only when g is in S, i.e., just x[g].  Actually: the term is x[g] * x[e]
    # but e is not in S so x[e] = 0. Wait -- let me reconsider.
    #
    # The 2-path count from e to h is:
    #   |{y in G : e->y and y->h}| = |{y : y in S and y^{-1}h in S}|
    #
    # Since e->y iff y in S (i.e., e^{-1}y = y in S), and y->h iff y^{-1}h in S.
    # So for y in G\{e}, this is x[y] * (1 if y^{-1}h in S else 0).
    #
    # Now y^{-1}h: if y = h, then y^{-1}h = e, which is never in S. So that
    # term is 0.  If y != h and y^{-1}h != e, then the term is x[y] * x[y^{-1}h].
    #
    # products[g][h] gives the non-id index of g^{-1}h, or -1 if g^{-1}h = e
    # (which happens when g = h).

    # Deduplicate product auxiliary variables.
    # For each product x[a] * x[b], we need at most one auxiliary per
    # unordered pair {a, b} (or just x[a] if a == b, since x[a]^2 = x[a]
    # for binary variables).
    product_aux: dict[tuple[int, int], gp.Var] = {}

    def _get_product_var(a: int, b: int) -> gp.Var:
        """Get or create the auxiliary variable for x[a] * x[b]."""
        if a == b:
            return x[a]
        key = (min(a, b), max(a, b))
        if key not in product_aux:
            p = model.addVar(vtype=GRB.BINARY, name=f"p_{key[0]}_{key[1]}")
            model.addGenConstrAnd(
                p,
                [x[key[0]], x[key[1]]],
                name=f"p_and_{key[0]}_{key[1]}",
            )
            product_aux[key] = p
        return product_aux[key]

    for h in range(m):
        terms: list[gp.Var] = []
        for g in range(m):
            prod_idx = products[g][h]
            if prod_idx == -1:
                # g^{-1}h = e, meaning g = h; y^{-1}h = e not in S, term is 0.
                continue
            terms.append(_get_product_var(g, prod_idx))

        model.addConstr(
            gp.quicksum(terms) == lam_minus_mu * x[h] + mu,
            name=f"dsrg_{h}",
        )

    # ── Automorphism pruning (lex-leader) ────────────────────────────────
    if use_aut_pruning and group_data.aut_generators:
        _add_aut_lex_leader(model, x, group_data.aut_generators)

    model.update()
    return model, x


# ── Lex-leader constraints for automorphism pruning ─────────────────────────


def _add_aut_lex_leader(
    model: gp.Model,
    x: list[gp.Var],
    aut_generators: list[list[int]],
) -> None:
    """Add lex-leader constraints for each Aut(G) generator.

    For each generator phi, we require that the binary vector
    (x[0], x[1], ..., x[m-1]) is lexicographically <= the permuted vector
    (x[phi(0)], x[phi(1)], ..., x[phi(m-1)]).

    Uses the standard MIP lex-leader encoding with agreement-tracking
    auxiliary binary variables.
    """
    m = len(x)

    for gen_idx, perm in enumerate(aut_generators):
        # Skip identity permutations.
        if all(perm[i] == i for i in range(m)):
            continue

        # Find the last position where the permutation differs from identity.
        # No need to track agreement beyond that point.
        last_diff = 0
        for j in range(m):
            if perm[j] != j:
                last_diff = j
        num_cols = last_diff + 1

        # g[j] = 1 means x[0..j] == x[perm(0..j)]  (agreement through column j).
        g = model.addVars(
            num_cols,
            vtype=GRB.BINARY,
            name=f"autlex_g_{gen_idx}",
        )

        for j in range(num_cols):
            xj = x[j]
            xpj = x[perm[j]]

            # g[j] = 1 => x[j] == x[perm[j]]
            model.addConstr(
                g[j] <= 1 - xj + xpj,
                name=f"autlex_eq_hi_{gen_idx}_{j}",
            )
            model.addConstr(
                g[j] <= 1 + xj - xpj,
                name=f"autlex_eq_lo_{gen_idx}_{j}",
            )

            # Chain: g[j] <= g[j-1]
            if j > 0:
                model.addConstr(
                    g[j] <= g[j - 1],
                    name=f"autlex_chain_{gen_idx}_{j}",
                )

            # Lex constraint: if all previous agree, x[j] <= x[perm[j]]
            if j == 0:
                model.addConstr(
                    xj - xpj <= 0,
                    name=f"autlex_lex_{gen_idx}_{j}",
                )
            else:
                model.addConstr(
                    xj - xpj <= 1 - g[j - 1],
                    name=f"autlex_lex_{gen_idx}_{j}",
                )


# ── Quadratic model builder ───────────────────────────────────────────────────


def build_cayley_dsrg_quad(
    n: int,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
    group_data: CayleyGroupData,
    *,
    use_aut_pruning: bool = True,
    undirected: bool = False,
    quiet: bool = True,
) -> tuple[gp.Model, list[gp.Var]]:
    """Quadratic formulation: only n-1 binary variables, no auxiliaries.

    Uses Gurobi's NonConvex=2 to handle bilinear x[g]*x[h] terms directly
    via spatial branching, avoiding the weak LP relaxations of the linearized
    AND formulation.

    Same constraints as build_cayley_dsrg but products are expressed as
    quadratic expressions instead of being linearized with auxiliary variables.
    """
    m = group_data.num_nonid
    products = group_data.products
    lam_minus_mu = lambda_param - mu

    model = gp.Model(
        f"CayleyDSRG_Q_{n}_{k}_{t}_{lambda_param}_{mu}_g{group_data.lib_id}"
    )
    model.setParam("NonConvex", 2)
    if quiet:
        model.setParam("OutputFlag", 0)

    model.setObjective(0, GRB.MINIMIZE)

    # ── Decision variables: x[g] for g in G\{e} ─────────────────────────
    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{g}") for g in range(m)]

    # ── Degree ────────────────────────────────────────────────────────────
    model.addConstr(gp.quicksum(x) == k, name="degree")

    # ── Undirected: S = S^{-1} ────────────────────────────────────────────
    if undirected:
        for g_lo, g_hi in group_data.pairs:
            model.addConstr(x[g_lo] == x[g_hi], name=f"sym_{g_lo}_{g_hi}")

    # ── Reciprocal count ──────────────────────────────────────────────────
    # Involutions contribute 1, complete pairs contribute 2.
    recip_expr = gp.QuadExpr()
    for g in group_data.involutions:
        recip_expr += x[g]
    for g_lo, g_hi in group_data.pairs:
        recip_expr += 2 * x[g_lo] * x[g_hi]
    model.addQConstr(recip_expr == t, name="reciprocal")

    # ── DSRG 2-path condition at identity ─────────────────────────────────
    for h in range(m):
        lhs = gp.QuadExpr()
        for g in range(m):
            prod_idx = products[g][h]
            if prod_idx == -1:
                continue
            if g == prod_idx:
                lhs += x[g]  # x[g]^2 = x[g] for binary
            else:
                lhs += x[g] * x[prod_idx]
        model.addQConstr(
            lhs == lam_minus_mu * x[h] + mu,
            name=f"dsrg_{h}",
        )

    # ── Automorphism pruning ──────────────────────────────────────────────
    if use_aut_pruning and group_data.aut_generators:
        _add_aut_lex_leader(model, x, group_data.aut_generators)

    model.update()
    return model, x


# ── Solve wrapper ────────────────────────────────────────────────────────────


def solve_cayley_dsrg(
    n: int,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
    lib_id: int,
    *,
    use_aut_pruning: bool = True,
    undirected: bool = False,
    quadratic: bool = False,
    threads: int = -1,
    time_limit: float | None = None,
    heuristics: float | None = None,
    quiet: bool = False,
    gurobi_params: dict[str, str] | None = None,
) -> dict:
    """Build and solve a Cayley-graph DSRG ILP for SmallGroup(n, lib_id).

    Args:
        n: Group order (= number of vertices).
        k: In- and out-degree.
        t: Reciprocal neighbours per vertex.
        lambda_param: lambda parameter.
        mu: mu parameter.
        lib_id: GAP SmallGroup library ID.
        use_aut_pruning: Use Aut(G) lex-leader symmetry breaking.
        threads: Solver threads (-1 = Gurobi default).
        time_limit: Wall-clock limit in seconds.
        heuristics: Fraction of time on MIP heuristics.
        quiet: Suppress Gurobi output.
        gurobi_params: Extra Gurobi parameter overrides.

    Returns:
        Dict with status, timing, parameters, and optionally the
        connection set S.
    """
    group_data = load_cayley_data(n, lib_id)

    builder = build_cayley_dsrg_quad if quadratic else build_cayley_dsrg
    model, x_vars = builder(
        n,
        k,
        t,
        lambda_param,
        mu,
        group_data,
        use_aut_pruning=use_aut_pruning,
        undirected=undirected,
        quiet=quiet,
    )

    if threads >= 0:
        model.setParam("Threads", threads)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if heuristics is not None:
        model.setParam("Heuristics", heuristics)
    model.setParam("MIPFocus", 1)

    for key, val in (gurobi_params or {}).items():
        for conv in (int, float):
            try:
                val = conv(val)  # type: ignore[assignment]
                break
            except ValueError:
                continue
        model.setParam(key, val)

    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0

    status = _extract_status(model)

    connection_set = None
    if model.SolCount > 0:
        connection_set = [
            g for g in range(group_data.num_nonid) if round(x_vars[g].X) == 1
        ]

    return {
        "status": status,
        "wall_seconds": round(elapsed, 4),
        "n": n,
        "k": k,
        "t": t,
        "lambda": lambda_param,
        "mu": mu,
        "lib_id": lib_id,
        "group_name": group_data.name,
        "use_aut_pruning": use_aut_pruning,
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs,
        "num_gen_constrs": model.NumGenConstrs,
        "connection_set": connection_set,
        "connection_set_original_indices": (
            [group_data.nonid_order[g] for g in connection_set]
            if connection_set is not None
            else None
        ),
    }


# ── Multi-group search ───────────────────────────────────────────────────────


def list_groups(n: int, *, include_abelian: bool = False) -> list[tuple[int, str]]:
    """Enumerate the groups of order n via GAP.

    Args:
        n: Group order.
        include_abelian: If True, list abelian groups too.

    Returns:
        List of ``(lib_id, structure_description)`` in GAP library order.
    """
    abelian_filter = "" if include_abelian else "not IsAbelian(G) and "
    gap_script = f"""
LoadPackage("smallgrp");;
groups := Filtered(AllSmallGroups({n}), G -> {abelian_filter}true);;
for Gi in groups do
    gid := IdGroup(Gi);;
    Print(gid[2], " ", StructureDescription(Gi), "\\n");
od;
Print("DONE\\n");
QUIT;
"""
    proc = subprocess.run(
        [GAP_BIN, "-q"],
        input=gap_script,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GAP failed: {proc.stderr}")

    groups: list[tuple[int, str]] = []
    for line in proc.stdout.strip().splitlines():
        if line.strip() == "DONE":
            break
        parts = line.strip().split(maxsplit=1)
        if parts:
            groups.append((int(parts[0]), parts[1] if len(parts) > 1 else "?"))
    return groups


def search_all_groups(
    n: int,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
    *,
    use_aut_pruning: bool = True,
    undirected: bool = False,
    threads: int = -1,
    time_limit: float | None = None,
    quiet: bool = False,
    gurobi_params: dict[str, str] | None = None,
    include_abelian: bool = False,
) -> list[dict]:
    """Search for Cayley DSRGs across all groups of order n.

    Uses GAP to enumerate groups, then solves the ILP for each.

    Args:
        include_abelian: If True, search abelian groups too.

    Returns:
        List of result dicts, one per group attempted.
    """
    groups = list_groups(n, include_abelian=include_abelian)

    results: list[dict] = []
    for lib_id, group_name in groups:
        if not quiet:
            print(f"  Group {lib_id}: {group_name} ... ", end="", flush=True)

        try:
            result = solve_cayley_dsrg(
                n,
                k,
                t,
                lambda_param,
                mu,
                lib_id,
                use_aut_pruning=use_aut_pruning,
                undirected=undirected,
                threads=threads,
                time_limit=time_limit,
                quiet=quiet,
                gurobi_params=gurobi_params,
            )
            results.append(result)

            if not quiet:
                cs = result["connection_set"]
                if result["status"] == "Optimal" and cs is not None:
                    print(f"FOUND |S|={len(cs)} in {result['wall_seconds']:.2f}s")
                elif result["status"] == "Infeasible":
                    print(f"infeasible in {result['wall_seconds']:.2f}s")
                else:
                    print(f"{result['status']} in {result['wall_seconds']:.2f}s")

        except Exception as e:
            if not quiet:
                print(f"ERROR: {e}")
            results.append(
                {
                    "status": f"Error: {e}",
                    "wall_seconds": 0,
                    "n": n,
                    "k": k,
                    "t": t,
                    "lambda": lambda_param,
                    "mu": mu,
                    "lib_id": lib_id,
                    "group_name": group_name,
                    "connection_set": None,
                }
            )

    return results
