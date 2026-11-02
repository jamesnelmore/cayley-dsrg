import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="DSRG results")


@app.cell
def _():
    import sqlite3
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    return Path, mo, np, pd, plt, sqlite3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cayley DSRG search — results

    Everything the paper reports, read from `data/dpds.sqlite`. Rebuild it from
    the source-of-truth CSVs before trusting anything here:

    ```
    uv run python scripts/build_database.py data
    ```

    The build recomputes group names, the candidate grid (`group_param`),
    canonical `digraph6` forms and graph invariants via GAP, and loads the
    run-log extracts in `data/logs/` (`scripts/extract_hpc_logs.py`).

    **Methods in `searches`**

    | method | what it is |
    |---|---|
    | `ilp-30` | ILP pass 1: every pair, 30 min per group, 10 cores |
    | `ilp-720` | ILP pass 2: pairs pass 1 left unsettled, 12 h each, 2 cores |
    | `rrhc-B` | RRHC sweep, restart budget $B \in \{10, 50, \dots, 10^5\}$, fixed seed 20260725, 32 cores |
    | `rrhc` | legacy unseeded 10k-restart RRHC runs (n ≥ 64); superseded by the sweep |
    | `exhaustive` | legacy GPU exhaustive search (small n) |

    The sweep's budgets are *nested*: restart $i$ uses seed $20260725 + i$ at
    every budget and the winner is the lowest-index solving restart, so the
    smallest budget at which a pair is solved brackets the index of its first
    successful restart.

    A *pair* is (nonabelian group $G$, open parameter set) with $|G| = n$.
    """)
    return


@app.cell
def _(Path, mo, pd, sqlite3):
    ROOT = Path(mo.notebook_dir() or Path.cwd()).resolve()
    DB = ROOT / "data" / "dpds.sqlite"
    if not DB.exists():
        raise FileNotFoundError(f"{DB} -- run scripts/build_database.py data")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    def q(sql: str, **params: object) -> pd.DataFrame:
        """Run read-only SQL against the results database."""
        return pd.read_sql_query(sql, con, params=params or None)

    KEY = ["n", "k", "t", "lambda", "mu"]
    PAIR = KEY + ["lib_id"]
    BUDGETS = [10, 50, 100, 500, 1000, 5000, 10000, 100000]

    mo.md(f"Connected read-only to `{DB}` (modified "
          f"{pd.Timestamp(DB.stat().st_mtime, unit='s'):%Y-%m-%d %H:%M} UTC).")
    return BUDGETS, KEY, PAIR, q


@app.cell
def _(mo, q):
    _logs = q(
        "SELECT (SELECT COUNT(*) FROM ilp_solves) AS ilp, "
        "(SELECT COUNT(*) FROM rrhc_best_error) AS err, "
        "(SELECT COUNT(*) FROM rrhc_task_times) AS times"
    ).iloc[0]
    mo.stop(
        _logs.sum() == 0,
        mo.callout(mo.md("The run-log tables are empty: rebuild with `data/logs/` present."), kind="danger"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building the pair table

    The universe is `group_param` restricted to open parameter sets. The sweep
    wrote a row for every pair it attempted, so every pair must have an
    `rrhc-10` row — asserted below.
    """)
    return


@app.cell
def _(BUDGETS, q):
    pairs = q("""
        WITH u AS (
            SELECT gp.group_id, gp.param_id, g.order_n AS n, g.lib_id, g.name,
                   p.k, p.t, p.lambda, p.mu
            FROM group_param gp
            JOIN groups g     ON g.group_id = gp.group_id
            JOIN parameters p ON p.param_id = gp.param_id
            WHERE p.existence_status = 'open'
        )
        SELECT u.*,
               MAX(s.outcome = 'found')                                    AS realized,
               MAX(s.outcome IN ('infeasible_proof', 'empty_proof'))       AS proved,
               MAX(CASE WHEN s.method = 'ilp-30'  THEN s.outcome END)      AS ilp30,
               MAX(CASE WHEN s.method = 'ilp-720' THEN s.outcome END)      AS ilp720,
               MAX(CASE WHEN s.method = 'rrhc'    THEN s.outcome END)      AS legacy_rrhc,
               MAX(s.method = 'rrhc-10')                                   AS swept,
               MIN(CASE WHEN s.method LIKE 'rrhc-%' AND s.outcome = 'found'
                        THEN CAST(SUBSTR(s.method, 6) AS INTEGER) END)     AS min_budget,
               GROUP_CONCAT(CASE WHEN s.outcome = 'found' THEN s.method END) AS found_by
        FROM u LEFT JOIN searches s
          ON s.group_id = u.group_id AND s.param_id = u.param_id
        GROUP BY u.group_id, u.param_id
        ORDER BY u.n, u.k, u.t, u.lambda, u.mu, u.lib_id
    """)
    assert (pairs["swept"] == 1).all(), "some pairs have no sweep row"
    _both = (pairs["realized"] == 1) & (pairs["proved"] == 1)
    assert not _both.any(), f"{_both.sum()} pairs both realized and proved infeasible"
    assert set(pairs["min_budget"].dropna().astype(int)) <= set(BUDGETS)

    pairs["status"] = "open"
    pairs.loc[pairs["proved"] == 1, "status"] = "infeasible"
    pairs.loc[pairs["realized"] == 1, "status"] = "realized"
    pairs["ilp"] = pairs["ilp720"].fillna(pairs["ilp30"])  # pass 2 supersedes pass 1
    return (pairs,)


@app.cell
def _(pd, q):
    param_status = q("""
        WITH pg AS (
            SELECT gp.param_id, gp.group_id,
                   MAX(s.outcome = 'found') AS realized,
                   MAX(s.outcome IN ('infeasible_proof', 'empty_proof')) AS proved
            FROM group_param gp
            LEFT JOIN searches s ON s.group_id = gp.group_id AND s.param_id = gp.param_id
            GROUP BY gp.param_id, gp.group_id
        )
        SELECT p.param_id, p.n, p.k, p.t, p.lambda, p.mu,
               COUNT(pg.group_id)                                   AS groups,
               COALESCE(SUM(pg.realized), 0)                        AS realized,
               COALESCE(SUM(pg.proved AND NOT pg.realized), 0)      AS infeasible,
               COALESCE(SUM(NOT (pg.realized OR pg.proved)), 0)     AS open,
               (SELECT COUNT(*) FROM graphs gr WHERE gr.param_id = p.param_id) AS graphs
        FROM parameters p
        LEFT JOIN pg ON pg.param_id = p.param_id
        WHERE p.existence_status = 'open'
        GROUP BY p.param_id
        ORDER BY p.n, p.k, p.t, p.lambda, p.mu
    """)

    def _cat(r: pd.Series) -> str:
        if r["groups"] == 0:
            return "abelian-only order"
        if r["realized"] > 0:
            return "realized"
        if r["open"] == 0:
            return "ruled out"
        return "open"

    param_status["category"] = param_status.apply(_cat, axis=1)
    return (param_status,)


@app.cell(hide_code=True)
def _(mo, pairs, param_status):
    _pc = pairs["status"].value_counts()
    _cc = param_status["category"].value_counts()
    _g = param_status.loc[param_status["category"] == "realized", "graphs"].sum()
    mo.md(f"""
    ## Headline

    | | count |
    |---|---|
    | open parameter sets ($n \\le {param_status['n'].max()}$) | **{len(param_status)}** |
    | &nbsp;&nbsp;realized by a Cayley DSRG | **{_cc.get('realized', 0)}** |
    | &nbsp;&nbsp;ruled out on every nonabelian group | **{_cc.get('ruled out', 0)}** |
    | &nbsp;&nbsp;order has only abelian groups (no Cayley DSRG, Klin et al.) | **{_cc.get('abelian-only order', 0)}** |
    | &nbsp;&nbsp;still open for Cayley graphs | **{_cc.get('open', 0)}** |
    | non-isomorphic DSRGs on previously open parameter sets | **{_g}** |
    | (group, parameter) pairs searched | **{len(pairs):,}** |
    | &nbsp;&nbsp;realized | **{_pc.get('realized', 0):,}** |
    | &nbsp;&nbsp;proved infeasible | **{_pc.get('infeasible', 0):,}** |
    | &nbsp;&nbsp;unresolved | **{_pc.get('open', 0):,}** |

    "Ruled out" here is computational only; the dihedral section below adds
    eliminations from He–Zhang–Feng.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. Provenance and coverage

    Rows per method restricted to the search space.
    """)
    return


@app.cell
def _(mo, q):
    _s = q("""
        SELECT s.method, s.outcome, COUNT(*) AS rows
        FROM searches s
        JOIN group_param gp ON gp.group_id = s.group_id AND gp.param_id = s.param_id
        JOIN parameters p   ON p.param_id = s.param_id
        WHERE p.existence_status = 'open'
        GROUP BY s.method, s.outcome
    """)
    _t = _s.pivot_table(index="method", columns="outcome", values="rows", fill_value=0)
    _t["pairs covered"] = _t.sum(axis=1)  # UNIQUE(group_id, param_id, method)
    _order = ["exhaustive", "ilp-30", "ilp-720", "rrhc"] + [
        f"rrhc-{b}" for b in [10, 50, 100, 500, 1000, 5000, 10000, 100000]
    ]
    method_table = _t.reindex([m for m in _order if m in _t.index]).astype(int)
    _outside = q("""
        SELECT COUNT(*) AS n FROM searches s
        LEFT JOIN group_param gp ON gp.group_id = s.group_id AND gp.param_id = s.param_id
        JOIN parameters p ON p.param_id = s.param_id
        WHERE gp.group_id IS NULL OR p.existence_status <> 'open'
    """)["n"][0]
    mo.vstack([
        mo.md(f"{_outside:,} search rows lie outside the search space "
              "(known/impossible parameter sets or abelian groups) and are ignored."),
        mo.ui.table(method_table.reset_index(), selection=None),
    ])
    return


@app.cell
def _(mo, pairs):
    _no_ilp = pairs[pairs["ilp"].isna()]
    _no_p1 = pairs[pairs["ilp30"].isna()]
    mo.md(f"""
    * **{len(_no_p1):,}** pairs have no pass-1 row: their pass-1 task hit the
      3-day wall clock (23 tasks) or failed (2), so they went straight to pass 2
      unless already solved.
    * **{len(_no_ilp):,}** pairs have no ILP row at all, of which
      **{int((_no_ilp['status'] == 'realized').sum())}** are realized (pass 2 skipped
      pairs RRHC had already solved). Unrealized pairs with no ILP row:
      **{int((_no_ilp['status'] != 'realized').sum())}**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B. Search space and outcomes by order
    """)
    return


@app.cell
def _(mo, pairs, param_status, pd, q):
    _p = param_status.groupby("n").agg(
        open_params=("category", "size"),
        abelian_only=("category", lambda c: (c == "abelian-only order").sum()),
        realized_params=("category", lambda c: (c == "realized").sum()),
        ruled_out_params=("category", lambda c: (c == "ruled out").sum()),
    )
    _q = pairs.groupby("n")["status"].value_counts().unstack(fill_value=0)
    _q = _q.reindex(columns=["realized", "infeasible", "open"], fill_value=0)
    _q.columns = ["realized_pairs", "infeasible_pairs", "open_pairs"]
    _g = q("SELECT order_n AS n, COUNT(*) AS nonabelian_groups FROM groups "
           "WHERE is_abelian = 0 GROUP BY order_n").set_index("n")
    by_n = _p.join(_g, how="left").join(_q, how="left").fillna(0).astype(int).reset_index()
    by_n["pairs"] = by_n[["realized_pairs", "infeasible_pairs", "open_pairs"]].sum(axis=1)
    by_n = by_n[[
        "n", "open_params", "abelian_only", "nonabelian_groups", "pairs", "realized_pairs",
        "infeasible_pairs", "open_pairs", "realized_params", "ruled_out_params",
    ]]
    _tot = by_n.drop(columns="n").sum().to_frame().T.assign(n="total")
    mo.ui.table(pd.concat([by_n, _tot], ignore_index=True), selection=None, page_size=80)
    return (by_n,)


@app.cell
def _(by_n, plt):
    COL = {
        "realized": "#2a78d6",    # categorical slot 1
        "infeasible": "#eb6834",  # categorical slot 2
        "open": "#c3c2b7",        # neutral: unresolved
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e4e3df",
    }
    plt.rcParams.update({
        "font.size": 9,
        "axes.edgecolor": COL["ink2"],
        "axes.labelcolor": COL["ink"],
        "xtick.color": COL["ink2"],
        "ytick.color": COL["ink2"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": COL["grid"],
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })

    def fig_outcomes_by_n() -> plt.Figure:
        d = by_n[by_n["pairs"] > 0].reset_index(drop=True)
        frac = d[["realized_pairs", "infeasible_pairs", "open_pairs"]].div(d["pairs"], axis=0)
        fig, ax = plt.subplots(figsize=(7.0, 2.6), layout="constrained")
        x = range(len(d))
        bottom = [0.0] * len(d)
        for col, lab, key in [
            ("realized_pairs", "realized", "realized"),
            ("infeasible_pairs", "proved infeasible", "infeasible"),
            ("open_pairs", "unresolved", "open"),
        ]:
            ax.bar(x, frac[col], bottom=bottom, color=COL[key], label=lab, width=0.8,
                   edgecolor="white", linewidth=0.6)
            bottom = [b + v for b, v in zip(bottom, frac[col])]
        ax.set_xticks(list(x), d["n"].astype(str), rotation=90, fontsize=7)
        ax.set_xlim(-0.6, len(d) - 0.4)
        ax.set_ylim(0, 1)
        ax.set_ylabel("share of pairs")
        ax.set_xlabel("n")
        ax.grid(axis="x", visible=False)
        ax.legend(ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
        return fig

    fig_outcomes_by_n()
    return COL, fig_outcomes_by_n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C. Parameter sets

    One row per open parameter set. `groups` is the number of nonabelian
    groups of order $n$; `graphs` the number of non-isomorphic DSRGs found.
    """)
    return


@app.cell
def _(mo):
    category_filter = mo.ui.dropdown(
        options=["all", "realized", "ruled out", "open", "abelian-only order"],
        value="all",
        label="category",
    )
    category_filter
    return (category_filter,)


@app.cell
def _(category_filter, mo, param_status):
    _v = category_filter.value
    _d = param_status if _v == "all" else param_status[param_status["category"] == _v]
    mo.ui.table(_d.drop(columns="param_id"), selection=None, page_size=25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## D. Constructions on open parameter sets

    Every realized pair: group, smallest sweep budget that solved it, every
    method that found it, stored connection sets, and how many non-isomorphic
    digraphs those connection sets give.
    """)
    return


@app.cell
def _(KEY, mo, pairs, q):
    _sets = q("""
        SELECT group_id, param_id, COUNT(*) AS connection_sets,
               COUNT(DISTINCT graph_id) AS graphs_on_group
        FROM dpds GROUP BY group_id, param_id
    """)
    constructions = (
        pairs[pairs["status"] == "realized"]
        .merge(_sets, on=["group_id", "param_id"], how="left")
        [KEY + ["lib_id", "name", "min_budget", "found_by", "ilp", "connection_sets", "graphs_on_group"]]
        .reset_index(drop=True)
    )
    constructions["min_budget"] = constructions["min_budget"].astype("Int64")
    mo.vstack([
        mo.md(
            f"**{len(constructions)}** realized pairs over "
            f"**{constructions[KEY].drop_duplicates().shape[0]}** parameter sets, "
            f"**{int(constructions['connection_sets'].sum())}** stored connection sets."
        ),
        mo.ui.table(constructions, selection=None, page_size=20),
    ])
    return (constructions,)


@app.cell
def _(mo, pd, q):
    _rows = q("""
        SELECT d.group_id, d.param_id, d.source_method, d.graph_id
        FROM dpds d
        JOIN parameters p ON p.param_id = d.param_id
        WHERE p.existence_status = 'open'
          AND EXISTS (SELECT 1 FROM searches s
                      WHERE s.group_id = d.group_id AND s.param_id = d.param_id
                        AND s.outcome = 'found')
    """)
    _tags = _rows["source_method"].map(lambda m: [x.strip() for x in m.split(",")])

    def _origin(tags: list[str]) -> str:
        """The method that first stored the set (later methods are appended)."""
        first = tags[0]
        if first.startswith("rrhc-"):
            return "RRHC sweep"
        if first == "rrhc":
            return "legacy RRHC (unseeded)"
        if first == "inverse":
            return "inverse S^-1 of an earlier find"
        if first.startswith("ilp"):
            return "ILP pass 1"
        return first

    _rows["origin"] = _tags.map(_origin)
    _rows["by_sweep"] = _tags.map(lambda t: any(x.startswith("rrhc-") for x in t))
    _rows["pair"] = list(zip(_rows["group_id"], _rows["param_id"]))
    _order = ["RRHC sweep", "legacy RRHC (unseeded)", "inverse S^-1 of an earlier find", "ILP pass 1"]
    set_origins = (
        _rows.groupby("origin")
        .agg(rows=("pair", "size"), pairs=("pair", "nunique"), also_returned_by_sweep=("by_sweep", "sum"))
        .reindex([o for o in _order if o in set(_rows["origin"])] + sorted(set(_rows["origin"]) - set(_order)))
        .reset_index()
    )
    _per_pair = _rows.groupby("pair").agg(sets=("pair", "size"), graphs=("graph_id", "nunique"))
    sets_per_pair = (
        pd.crosstab(_per_pair["sets"], _per_pair["graphs"], margins=True, margins_name="total")
        .rename_axis(index="sets stored", columns=None)
    )
    sets_per_pair.columns = [
        c if c == "total" else f"{c} graph{'s' if c != 1 else ''}" for c in sets_per_pair.columns
    ]
    _n_pairs = _rows["pair"].nunique()
    _sweep_rows = int(_rows["by_sweep"].sum())
    def _get(origin: str, col: str) -> int:
        return int(set_origins.loc[set_origins["origin"] == origin, col].sum())

    _legacy_dup = int(
        (_rows[_rows["origin"] == "legacy RRHC (unseeded)"].groupby("pair").size() > 1).sum()
    )
    mo.vstack([
        mo.md(f"""
        ### Why {len(_rows)} stored connection sets for {_n_pairs} realized pairs

        A *pair* is realized as soon as one DPDS is known; `dpds` keeps **every
        distinct connection set** any method stored for it, so the two counts
        measure different things.

        * **The sweep stores exactly one set per pair.** Restart seeds are fixed and
          the lowest-index solving restart wins, so every budget returns the same set.
          That accounts for **{_sweep_rows}** sets, one per realized pair; of these,
          {_sweep_rows - _get('RRHC sweep', 'rows')} coincided with a set an earlier
          method had already stored, so they appear under that method below.
        * **Everything else is left over from earlier runs:**
          **{_get('legacy RRHC (unseeded)', 'rows')}** sets from the unseeded legacy
          RRHC runs ({_get('legacy RRHC (unseeded)', 'pairs')} pairs; {_legacy_dup} of them
          have two sets because two runs were merged under the one `rrhc` label),
          **{_get('inverse S^-1 of an earlier find', 'rows')}** inverses $S^{{-1}}$ of
          earlier finds, added after the fact by an older script (a DPDS with the same
          parameters whose Cayley graph is the *converse*), and **{_get('ILP pass 1', 'rows')}** ILP solutions
          (lex-minimal under $\\mathrm{{Aut}}(G)$, so generally a different set from
          RRHC's even when the graph is the same).
        * **Different sets need not be different graphs.** Two sets on the same group
          give isomorphic digraphs when they're related by an automorphism of $G$
          (or, more rarely, by a non-automorphic isomorphism), and $S^{{-1}}$ gives the
          converse, which is isomorphic only when the graph is self-converse.

        So the number of stored sets per pair records **how many runs happened to
        touch the pair**, not how many solutions exist; it should not be reported as
        a count of DPDSs. The meaningful counts are realized pairs and
        non-isomorphic graphs.
        """),
        mo.ui.table(set_origins, selection=None),
        mo.md("Stored sets per realized pair against the distinct graphs they give:"),
        mo.ui.table(sets_per_pair.reset_index(), selection=None),
    ])
    return


@app.cell
def _(mo, q):
    graphs_open = q("""
        SELECT gr.graph_id, gr.n, p.k, p.t, p.lambda, p.mu, gr.aut_order, gr.is_drr,
               gr.is_self_converse, gr.num_dpds, gr.num_groups, gr.digraph6
        FROM graphs gr JOIN parameters p ON p.param_id = gr.param_id
        WHERE p.existence_status = 'open'
        ORDER BY gr.n, p.k, p.t, p.lambda, p.mu, gr.graph_id
    """)
    _sc = int(graphs_open["is_self_converse"].sum())
    mo.vstack([
        mo.md(f"""
        **{len(graphs_open)}** non-isomorphic DSRGs on previously open parameter sets.
        **{int(graphs_open['is_drr'].sum())}** are digraphical regular representations
        ($|\\mathrm{{Aut}}| = n$); **{_sc}** are isomorphic to their converse
        ($S \\leftrightarrow S^{{-1}}$). The converse of each of the other
        {len(graphs_open) - _sc} is also a DSRG with the same parameters, but it is
        only in this table if its connection set was stored, so counting classes
        up to converse needs the converses canonicalized too.
        """),
        mo.ui.table(graphs_open, selection=None, page_size=15),
    ])
    return (graphs_open,)


@app.cell
def _(KEY, constructions, param_status, pd):
    def _summ(d: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "groups_realized": len(d),
            "min_restarts": int(d["min_budget"].min()),
            "group_ids": ", ".join(str(i) for i in sorted(d["lib_id"])),
        })

    realized_table = (
        constructions.groupby(KEY).apply(_summ, include_groups=False).reset_index()
        .merge(param_status[KEY + ["groups", "graphs"]], on=KEY)
    )
    realized_table
    return (realized_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E. ILP

    `ilp` is the final ILP verdict (pass 2 where it ran, else pass 1). Solve
    times are from the slurm logs (`ilp_solves`).
    """)
    return


@app.cell
def _(mo, pairs, pd):
    _p1 = pairs["ilp30"].value_counts(dropna=False).rename("pass 1 (30 min)")
    _p2 = pairs["ilp720"].value_counts(dropna=False).rename("pass 2 (12 h)")
    _t = pd.concat([_p1, _p2], axis=1).fillna(0).astype(int)
    _t.index = _t.index.fillna("no row")
    mo.ui.table(_t.reset_index(names="outcome"), selection=None)
    return


@app.cell
def _(mo, pairs):
    _t = pairs.groupby("n").agg(
        pairs=("ilp", "size"),
        p1_infeasible=("ilp30", lambda s: (s == "infeasible_proof").sum()),
        p1_timeout=("ilp30", lambda s: (s == "timeout").sum()),
        p1_found=("ilp30", lambda s: (s == "found").sum()),
        p1_no_row=("ilp30", lambda s: s.isna().sum()),
        p2_infeasible=("ilp720", lambda s: (s == "infeasible_proof").sum()),
        p2_timeout=("ilp720", lambda s: (s == "timeout").sum()),
        final_timeout_rate=("ilp", lambda s: round((s == "timeout").mean(), 3)),
    ).reset_index()
    mo.ui.table(_t, selection=None, page_size=60)
    return


@app.cell
def _(q):
    ilp_solves = q("SELECT method, status, seconds FROM ilp_solves")
    return (ilp_solves,)


@app.cell
def _(COL, ilp_solves, np, plt):
    def fig_ilp_times() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.5, 2.8), layout="constrained")
        for method, color, limit, lab in [
            ("ilp-30", COL["realized"], 1800, "pass 1 (30 min limit)"),
            ("ilp-720", COL["infeasible"], 43200, "pass 2 (12 h limit)"),
        ]:
            s = np.sort(ilp_solves.loc[
                (ilp_solves["method"] == method) & (ilp_solves["status"] == "Infeasible"), "seconds"
            ].to_numpy())
            s = np.maximum(s, 0.05)  # 0.0s rounds; keep on the log axis
            y = np.arange(1, len(s) + 1)
            ax.step(s, y, where="post", color=color, lw=2, label=f"{lab}: {len(s):,} pairs")
            ax.axvline(limit, color=color, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("time to prove infeasible (s; < 0.05 s plotted at 0.05)")
        ax.set_ylabel("pairs proved (cumulative)")
        ax.legend(loc="upper left", fontsize=8)
        return fig

    fig_ilp_times()
    return (fig_ilp_times,)


@app.cell
def _(ilp_solves, mo):
    mo.ui.table(
        ilp_solves.groupby(["method", "status"])["seconds"]
        .describe(percentiles=[0.5, 0.9, 0.99])
        .round(1)
        .reset_index(),
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F. RRHC restart-budget sweep
    """)
    return


@app.cell
def _(BUDGETS, KEY, mo, pairs, pd):
    _rows = []
    for _b in BUDGETS:
        _f = pairs[pairs["min_budget"] <= _b]
        _rows.append({
            "budget": _b,
            "pairs solved first here": int((pairs["min_budget"] == _b).sum()),
            "pairs realized (cumulative)": len(_f),
            "parameter sets realized (cumulative)": len(_f[KEY].drop_duplicates()),
        })
    budget_curve = pd.DataFrame(_rows)
    mo.ui.table(budget_curve, selection=None)
    return (budget_curve,)


@app.cell
def _(COL, budget_curve, plt):
    def fig_budget_curve() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), layout="constrained")
        for ax, col, lab in [
            (axes[0], "pairs realized (cumulative)", "(group, parameter) pairs realized"),
            (axes[1], "parameter sets realized (cumulative)", "parameter sets realized"),
        ]:
            ax.plot(budget_curve["budget"], budget_curve[col], color=COL["realized"], lw=2,
                    marker="o", ms=4)
            ax.set_xscale("log")
            ax.set_xlabel("restart budget per candidate")
            ax.set_title(lab, fontsize=9, loc="left")
            ax.set_ylim(bottom=0)
        return fig

    fig_budget_curve()
    return (fig_budget_curve,)


@app.cell
def _(BUDGETS, COL, np, pairs, plt):
    def fig_min_budget_vs_n() -> plt.Figure:
        d = pairs[pairs["min_budget"].notna()]
        rng = np.random.default_rng(0)
        level = d["min_budget"].map({b: i for i, b in enumerate(BUDGETS)}).to_numpy()
        fig, ax = plt.subplots(figsize=(6.5, 2.6), layout="constrained")
        ax.scatter(d["n"] + rng.uniform(-0.6, 0.6, len(d)), level + rng.uniform(-0.25, 0.25, len(d)),
                   s=10, color=COL["realized"], alpha=0.6, linewidths=0)
        ax.set_yticks(range(len(BUDGETS)), [f"{b:,}" for b in BUDGETS])
        ax.set_ylabel("smallest solving budget")
        ax.set_xlabel("n")
        return fig

    fig_min_budget_vs_n()
    return (fig_min_budget_vs_n,)


@app.cell
def _(mo, q):
    rrhc_cost = q("""
        SELECT CAST(SUBSTR(method, 6) AS INTEGER) AS budget,
               COUNT(*)                            AS tasks,
               ROUND(SUM(elapsed_s * ncpus) / 3600.0) AS core_hours,
               ROUND(MAX(elapsed_s) / 3600.0, 2)   AS max_task_h,
               GROUP_CONCAT(DISTINCT ncpus)        AS cpus
        FROM rrhc_task_times GROUP BY method ORDER BY budget
    """)
    mo.vstack([
        mo.md("Wall clock per parameter-set task (one task sweeps every nonabelian group "
              "of that order), from `sacct`:"),
        mo.ui.table(rrhc_cost, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Near misses

    For unresolved pairs, the best objective RRHC reached at $10^5$ restarts.
    The objective is $\sum_g r_g^2$ over the residual group-ring element
    $A^2 - tI - \lambda A - \mu(J - I - A)$, so 0 means a DPDS. Small values
    are candidates for a longer targeted run, not evidence of existence.
    """)
    return


@app.cell
def _(PAIR, mo, pairs, q):
    _be = q("SELECT group_id, param_id, best_error FROM rrhc_best_error WHERE method = 'rrhc-100000'")
    near = pairs[pairs["status"] == "open"].merge(_be, on=["group_id", "param_id"], how="left")
    _hist = near["best_error"].value_counts().sort_index().rename("open pairs").reset_index()
    mo.vstack([
        mo.md("Distribution of best error over unresolved pairs:"),
        mo.ui.table(_hist, selection=None, page_size=15),
        mo.md("Unresolved pairs with best error ≤ 4:"),
        mo.ui.table(
            near[near["best_error"] <= 4][PAIR + ["name", "best_error", "ilp"]]
            .sort_values(["best_error"] + PAIR),
            selection=None,
            page_size=15,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## G. ILP vs RRHC

    Final ILP verdict against the sweep, per pair. "RRHC solved" means some
    budget up to $10^5$ found a DPDS.
    """)
    return


@app.cell
def _(mo, pairs, pd):
    _x = pd.crosstab(
        pairs["ilp"].fillna("no ILP row"),
        pairs["min_budget"].notna().map({True: "RRHC solved", False: "RRHC none"}),
        margins=True,
    )
    _ilp_found = pairs[pairs["ilp"] == "found"]
    _rr_found = pairs[pairs["min_budget"].notna()]
    mo.vstack([
        mo.ui.table(_x.reset_index(names="ILP verdict"), selection=None),
        mo.md(f"""
        * ILP constructed **{len(_ilp_found)}** pairs; the sweep also solved
          **{int(_ilp_found['min_budget'].notna().sum())}** of them.
        * The sweep solved **{len(_rr_found)}** pairs; ILP had timed out on
          **{int((_rr_found['ilp'] == 'timeout').sum())}** of them and has no row for
          **{int(_rr_found['ilp'].isna().sum())}**.
        * Pass 2 (12 h) constructed **{int((pairs['ilp720'] == 'found').sum())}** pairs.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## H. Nearly-closed parameter sets

    Unrealized parameter sets with 1–9 nonabelian groups still unresolved.
    """)
    return


@app.cell
def _(KEY, mo, pairs, param_status):
    _nc = param_status[(param_status["category"] == "open") & (param_status["open"] < 10)]
    nearly_closed = (
        pairs[pairs["status"] == "open"]
        .merge(_nc[KEY + ["open", "groups"]], on=KEY)
        .sort_values(["open"] + KEY + ["lib_id"])[KEY + ["open", "groups", "lib_id", "name"]]
        .reset_index(drop=True)
    )
    mo.vstack([
        mo.md(f"**{len(_nc)}** parameter sets, **{len(nearly_closed)}** open groups in total."),
        mo.ui.table(nearly_closed, selection=None, page_size=20),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dihedral groups (He, Zhang & Feng, arXiv:1801.04748)

    $D_{2m} = \langle x, a \mid x^m = a^2 = 1, axa = x^{-1}\rangle$; every
    connection set is $x^X \cup x^Y a$.

    * **Theorems 1.1 + 1.2** (with Klin et al.): for $m = p$ an odd prime,
      $\mathrm{Cay}(D_{2p}, S)$ is a DSRG iff its parameters are
      $\bigl(2p,\, p-1+\varepsilon,\, \tfrac{p-1}{2}+\varepsilon,\,
      \tfrac{p-3}{2}+\varepsilon,\, \tfrac{p-1}{2}+\varepsilon\bigr)$,
      $\varepsilon \in \{0, 1\}$ (translated from their $(n,k,\mu,\lambda,t)$
      order). Any other parameter set is impossible on $D_{2p}$.
    * **Constructions 5.1–5.4** hold for every $m$ and are exhaustive within
      their connection-set shapes for prime-power $m$ (Theorems 6.1, 6.3, 7.2);
      they can only *realize* pairs, and only narrow what a construction on
      $D_{2p^\alpha}$ must look like.

    Parameters of every construction are recomputed from the adjacency matrix
    rather than read off the paper's formulas.
    """)
    return


@app.cell
def _(np):
    def isprime(x: int) -> bool:
        return x > 1 and all(x % d for d in range(2, int(x**0.5) + 1))

    def dih_mul(g: tuple[int, int], h: tuple[int, int], m: int) -> tuple[int, int]:
        """(x^i a^e)(x^j a^f) in D_2m."""
        i, e = g
        j, f = h
        return ((i + j) % m, f) if e == 0 else ((i - j) % m, (1 + f) % 2)

    def dih_params(m: int, X: list[int], Y: list[int]) -> tuple[int, ...] | None:
        """(n,k,t,lambda,mu) of Cay(D_2m, x^X u x^Y a), or None if not a DSRG."""
        els = [(i, e) for e in (0, 1) for i in range(m)]
        pos = {g: idx for idx, g in enumerate(els)}
        S = {(i % m, 0) for i in X} | {(j % m, 1) for j in Y}
        if (0, 0) in S:
            return None
        N = 2 * m
        A = np.zeros((N, N), dtype=np.int64)
        for g in els:
            for s in S:
                A[pos[g], pos[dih_mul(g, s, m)]] = 1
        A2 = A @ A
        deg = set(A.sum(1).tolist())
        if len(deg) != 1 or not (np.diag(A2) == A2[0, 0]).all():
            return None
        off = ~np.eye(N, dtype=bool)
        lam = set(A2[off & (A == 1)].tolist())
        mu = set(A2[off & (A == 0)].tolist())
        if len(lam) > 1 or len(mu) > 1:
            return None
        k = deg.pop()
        return (N, k, int(A2[0, 0]), lam.pop() if lam else 0, mu.pop() if mu else 0)

    def dih_constructions(m: int):
        """(name, v, X, Y) for Constructions 5.1-5.4 over divisors v of m."""
        for v in range(2, m + 1):
            if m % v:
                continue

            def lift(H, v=v):
                return sorted({(h + v * c) % m for h in H for c in range(m // v)})

            if v % 2:
                X1 = lift(range(1, (v + 1) // 2))
                yield ("5.1", v, X1, X1)
                Y3 = lift(range(0, (v + 1) // 2))
                yield ("5.3", v, [y for y in Y3 if y != 0], Y3)
                X4 = [y for y in Y3 if y % v]
                yield ("5.4", v, X4, sorted(set(X4) | set(lift([0]))))
            elif v > 2:
                X2 = lift(range(1, v // 2 + 1))
                yield ("5.2", v, X2, X2)

    def dih_prime_allowed(p: int) -> set[tuple[int, ...]]:
        return {
            (2 * p, p - 1 + e, (p - 1) // 2 + e, (p - 3) // 2 + e, (p - 1) // 2 + e)
            for e in (0, 1)
        }

    return dih_constructions, dih_params, dih_prime_allowed, isprime


@app.cell
def _(KEY, dih_prime_allowed, isprime, mo, pairs, param_status):
    _is_dih = pairs["name"].str.fullmatch(r"D\d+")
    dih = pairs[_is_dih].copy()
    dih["m"] = dih["n"] // 2
    _prime = dih[(dih["status"] == "open") & dih["m"].map(isprime)].copy()
    _prime["allowed"] = [
        tuple(r) in dih_prime_allowed(m) for r, m in zip(_prime[KEY].itertuples(index=False), _prime["m"])
    ]
    dih_eliminated = _prime[~_prime["allowed"]].merge(
        param_status[KEY + ["open", "realized"]].rename(
            columns={"open": "param_open", "realized": "param_realized"}
        ),
        on=KEY,
    )

    # sanity: every realized dihedrant on D_2p must have an allowed parameter set
    _real = dih[(dih["status"] == "realized") & dih["m"].map(isprime)]
    _viol = [
        r for r, m in zip(_real[KEY].itertuples(index=False), _real["m"])
        if tuple(r) not in dih_prime_allowed(m)
    ]
    _closes = dih_eliminated[
        (dih_eliminated["param_open"] == 1) & (dih_eliminated["param_realized"] == 0)
    ]
    dih_closed_params = _closes[KEY].drop_duplicates()
    mo.vstack([
        mo.md(f"""
        * Dihedral pairs in the search space: **{len(dih)}**
          ({int((dih['status'] == 'open').sum())} unresolved).
        * Unresolved pairs on $D_{{2p}}$ eliminated by Thm 1.2: **{len(dih_eliminated)}**.
        * Of those, the dihedral group was the *last* open group for
          **{len(dih_closed_params)}** parameter sets, which are therefore ruled out
          for Cayley graphs entirely.
        * Sanity check — realized dihedrants on $D_{{2p}}$ violating Thm 1.1:
          **{len(_viol)}** {'✅' if not _viol else '⚠️ ' + str(_viol)}
        """),
        mo.ui.table(
            dih_eliminated[KEY + ["name", "param_open", "param_realized"]], selection=None, page_size=15
        ),
    ])
    return dih, dih_closed_params


@app.cell
def _(KEY, dih, dih_constructions, dih_params, mo):
    _hits = []
    _cache: dict[int, dict] = {}
    for _, _r in dih[dih["status"] == "open"].iterrows():
        _m = int(_r["n"]) // 2
        if _m not in _cache:
            _cache[_m] = {}
            for _nm, _v, _X, _Y in dih_constructions(_m):
                _p = dih_params(_m, _X, _Y)
                if _p:
                    _cache[_m].setdefault(_p, (_nm, _v))
        _key = tuple(int(_r[c]) for c in KEY)
        if _key in _cache[_m]:
            _hits.append((_key, _r["name"], _cache[_m][_key]))
    mo.md(
        f"Constructions 5.1–5.4 checked on {len(_cache)} values of $m$: "
        f"**{len(_hits)}** unresolved dihedral pairs realized by them."
        + ("" if not _hits else f"\n\n{_hits}")
    )
    return


@app.cell(hide_code=True)
def _(dih_closed_params, mo, param_status):
    _cc = param_status["category"].value_counts()
    mo.md(f"""
    **Combined:** {_cc.get('ruled out', 0)} parameter sets ruled out
    computationally + {len(dih_closed_params)} closed by the dihedral theorem =
    **{_cc.get('ruled out', 0) + len(dih_closed_params)}** open parameter sets with
    no Cayley realization on a nonabelian group, plus
    {_cc.get('abelian-only order', 0)} on orders with only abelian groups.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## I. Fact-check against the current draft

    Claims in `dsrg-paper/main.tex` (draft of 2026-08-30) next to the value
    computed here.
    """)
    return


@app.cell
def _(
    constructions,
    dih_closed_params,
    graphs_open,
    mo,
    pairs,
    param_status,
    pd,
    q,
):
    _cc = param_status["category"].value_counts()
    _all_pairs = q("SELECT COUNT(*) AS c FROM group_param")["c"][0]
    _max_n = q("SELECT MAX(n) AS m FROM parameters")["m"][0]
    _sc = int(graphs_open["is_self_converse"].sum())
    fact_check = pd.DataFrame([
        ("Intro", "group-parameter set pairs to test", "37,671",
         f"{len(pairs):,} open; {_all_pairs:,} over all parameter sets"),
        ("Intro", "Hobart–Brouwer table / search range", "n ≤ 108", f"n ≤ {_max_n}"),
        ("Intro", "strongly regular Cayley digraphs constructed", "161",
         f"{len(graphs_open)} non-isomorphic ({_sc} self-converse) "
         f"from {int(constructions['connection_sets'].sum())} connection sets on {len(constructions)} pairs"),
        ("Intro", "previously-open parameter sets realized", "30", str(_cc.get("realized", 0))),
        ("Intro", "open parameter sets not realizable by Cayley graphs", "86",
         f"{_cc.get('ruled out', 0)} computational (+{len(dih_closed_params)} dihedral, "
         f"+{_cc.get('abelian-only order', 0)} abelian-only orders)"),
        ("§4", "first-pass ILP eliminated XX% of pairs",
         "XX", f"{(pairs['ilp30'] == 'infeasible_proof').mean():.1%} of all pairs"),
        ("§4", "first-pass ILP constructions", "XX", str(int((pairs["ilp30"] == "found").sum()))),
        ("§5", "RRHC first pass found a construction in XX% of cases", "XX",
         f"r=10: {(pairs['min_budget'] == 10).mean():.2%}; r=1e5: {pairs['min_budget'].notna().mean():.2%}"),
    ], columns=["section", "claim", "draft", "computed"])
    mo.ui.table(fact_check, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## J. Export to the paper

    Writes PDFs to `<paper>/figures/` and booktabs tables to
    `<paper>/tables/`. Nothing is written until you press the button.
    """)
    return


@app.cell
def _(Path, mo):
    paper_dir = mo.ui.text(
        value=str(Path.home() / "manuscripts" / "dsrg-paper"), label="paper directory", full_width=True
    )
    export = mo.ui.run_button(label="Export figures and tables")
    mo.vstack([paper_dir, export])
    return export, paper_dir


@app.function
def booktabs(df, cols: list[tuple[str, str]], align: str, caption: str, label: str) -> str:
    head = " & ".join(h for _, h in cols) + r" \\"
    body = "\n".join(
        "    " + " & ".join(str(r[c]) for c, _ in cols) + r" \\" for _, r in df.iterrows()
    )
    return (
        "% generated by combo-dl/results.py -- do not edit by hand\n"
        "\\begin{table}[ht]\n  \\centering\\small\n"
        f"  \\caption{{{caption}}}\\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{align}}}\n  \\toprule\n    {head}\n  \\midrule\n"
        f"{body}\n  \\bottomrule\n  \\end{{tabular}}\n\\end{{table}}\n"
    )


@app.cell
def _(KEY, by_n, param_status, realized_table):
    def _ptuple(r) -> str:
        return "$(" + ",".join(str(int(r[c])) for c in KEY) + ")$"

    _rt = realized_table.copy()
    _rt["param"] = _rt.apply(_ptuple, axis=1)
    _rt["frac"] = _rt["groups_realized"].astype(str) + "/" + _rt["groups"].astype(str)
    _rt["restarts"] = _rt["min_restarts"].map(lambda b: f"{b:,}".replace(",", "{,}"))
    _rt["group_ids"] = [
        ids if g <= 12 else f"\\emph{{{g} groups; see data}}"
        for ids, g in zip(_rt["group_ids"], _rt["groups_realized"])
    ]
    tex_realized = booktabs(
        _rt, [("param", "$(n,k,t,\\lambda,\\mu)$"), ("frac", "groups"), ("graphs", "graphs"),
              ("restarts", "min.\\ restarts"), ("group_ids", "SmallGroup ids")],
        "lrrrp{5.5cm}",
        "Previously open parameter sets realized as Cayley digraphs. "
        "\\emph{groups}: realizing / nonabelian groups of order $n$; "
        "\\emph{graphs}: non-isomorphic DSRGs found.",
        "tab:realized",
    )

    _ro = param_status[param_status["category"] == "ruled out"].copy()
    _ro["param"] = _ro.apply(_ptuple, axis=1)
    tex_ruled_out = booktabs(
        _ro, [("param", "$(n,k,t,\\lambda,\\mu)$"), ("groups", "groups ruled out")], "lr",
        "Open parameter sets with no Cayley realization on any nonabelian group of order $n$.",
        "tab:ruledout",
    )

    _bn = by_n[by_n["pairs"] > 0]
    tex_by_n = booktabs(
        _bn, [("n", "$n$"), ("open_params", "params"), ("nonabelian_groups", "groups"),
              ("pairs", "pairs"), ("realized_pairs", "realized"),
              ("infeasible_pairs", "infeasible"), ("open_pairs", "open")],
        "rrrrrrr", "Search outcomes by order.", "tab:byn",
    )
    tables = {
        "realized_params.tex": tex_realized,
        "ruled_out_params.tex": tex_ruled_out,
        "outcomes_by_n.tex": tex_by_n,
    }
    return (tables,)


@app.cell
def _(
    Path,
    export,
    fig_budget_curve,
    fig_ilp_times,
    fig_min_budget_vs_n,
    fig_outcomes_by_n,
    mo,
    paper_dir,
    plt,
    tables,
):
    mo.stop(not export.value, mo.md("_Press the button to export._"))
    _root = Path(paper_dir.value).expanduser()
    (_root / "figures").mkdir(parents=True, exist_ok=True)
    (_root / "tables").mkdir(parents=True, exist_ok=True)
    _written = []
    for _name, _fn in [
        ("outcomes_by_n.pdf", fig_outcomes_by_n),
        ("rrhc_budget_curve.pdf", fig_budget_curve),
        ("rrhc_min_budget_vs_n.pdf", fig_min_budget_vs_n),
        ("ilp_solve_times.pdf", fig_ilp_times),
    ]:
        _f = _fn()
        _f.savefig(_root / "figures" / _name)
        plt.close(_f)
        _written.append(f"figures/{_name}")
    for _name, _tex in tables.items():
        (_root / "tables" / _name).write_text(_tex)
        _written.append(f"tables/{_name}")
    mo.md("Wrote to `" + str(_root) + "`:\n\n" + "\n".join(f"* `{w}`" for w in _written))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## K. The sweep at 100,000 restarts

    What the sweep's largest budget ($10^5$ restarts) delivered on previously
    open parameter sets. Because the budgets are nested, `rrhc-100000` is the
    cumulative sweep result, and its one set per realized pair is tagged with
    that budget in `source_method`; every count below is this subset of `dpds`.
    These are lower bounds — the search samples, it does not enumerate.
    """)
    return


@app.cell
def _(BUDGETS, mo, q):
    _tag = f"%rrhc-{BUDGETS[-1]}%"
    sweep100 = q(
        """
        SELECT g.order_n AS n, p.k, p.t, p.lambda, p.mu, g.lib_id, g.name,
               d.graph_id, gr.aut_order, gr.is_drr, gr.is_self_converse, d.members
        FROM dpds d
        JOIN parameters p ON p.param_id = d.param_id
        JOIN groups g     ON g.group_id = d.group_id
        JOIN graphs gr    ON gr.graph_id = d.graph_id
        WHERE p.existence_status = 'open' AND d.source_method LIKE :tag
        ORDER BY n, k, t, lambda, mu, lib_id
        """,
        tag=_tag,
    )
    sweep100_graphs = q(
        """
        SELECT gr.graph_id, gr.n, p.k, p.t, p.lambda, p.mu, gr.aut_order,
               gr.is_drr, gr.is_self_converse,
               COUNT(*)      AS sweep_sets,
               gr.num_dpds   AS all_dpds,
               gr.digraph6
        FROM dpds d
        JOIN parameters p ON p.param_id = d.param_id
        JOIN graphs gr    ON gr.graph_id = d.graph_id
        WHERE p.existence_status = 'open' AND d.source_method LIKE :tag
        GROUP BY gr.graph_id
        ORDER BY gr.n, p.k, p.t, p.lambda, p.mu, gr.graph_id
        """,
        tag=_tag,
    )
    # aut_order is TEXT in the schema (orders elsewhere overflow int64); every
    # value in this subset fits, so cast it so the table sorts numerically.
    sweep100["aut_order"] = sweep100["aut_order"].astype("int64")
    sweep100_graphs["aut_order"] = sweep100_graphs["aut_order"].astype("int64")
    _key = ["n", "k", "t", "lambda", "mu"]
    _n_sets = len(sweep100)
    _n_params = sweep100[_key].drop_duplicates().shape[0]
    _n_graphs = len(sweep100_graphs)
    _n_drr = int(sweep100_graphs["is_drr"].sum())
    _n_sc = int(sweep100_graphs["is_self_converse"].sum())
    # the sweep stores exactly one set per realized pair
    assert sweep100.groupby(_key + ["lib_id"]).size().max() == 1, "pair with >1 sweep set"
    mo.vstack([
        mo.md(f"""
        | at $10^5$ restarts | count |
        |---|---|
        | DPDSs found (one connection set per realized pair) | **{_n_sets}** |
        | previously open parameter sets realized | **{_n_params}** |
        | non-isomorphic Cayley digraphs generated | **{_n_graphs}** |
        | &nbsp;&nbsp;of which digraphical regular representations ($\\lvert\\mathrm{{Aut}}\\rvert = n$) | **{_n_drr}** |
        | &nbsp;&nbsp;of which self-converse ($S \\cong S^{{-1}}$) | **{_n_sc}** |

        The {_n_sets} connection sets fall into {_n_graphs} isomorphism classes
        of digraph because distinct sets on the same group can be
        $\\mathrm{{Aut}}(G)$-equivalent (same graph), and the same graph recurs
        across groups and parameter sets.
        """),
        mo.md(f"**Every connection set found at $10^5$ restarts** ({_n_sets} rows):"),
        mo.ui.table(sweep100, selection=None, page_size=20),
        mo.md(f"**Every Cayley digraph they generate** ({_n_graphs} rows); "
              "`sweep_sets` = how many of the sets above land on it, "
              "`all_dpds` = total stored constructions mapping to it from every method:"),
        mo.ui.table(sweep100_graphs, selection=None, page_size=20),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## L. ILP nonexistence results

    The flip side of section K: where the ILP proves no DPDS exists. A pair's
    final ILP verdict is pass 2 where it ran, else pass 1; `infeasible_proof`
    means Gurobi certified the model empty. Gurobi's infeasibility is a
    floating-point verdict, not a formal certificate — see section E. Because
    abelian groups never carry a genuinely directed Cayley DSRG (Klin et al.),
    a parameter set with every nonabelian group of its order proved infeasible
    and none realized admits no Cayley DSRG at all.
    """)
    return


@app.cell
def _(KEY, mo, pairs, param_status):
    # pair-level: final ILP verdict, split by the pass that proved it
    _pairs_inf = int((pairs["ilp"] == "infeasible_proof").sum())
    _p1 = int((pairs["ilp30"] == "infeasible_proof").sum())
    _p2 = int((pairs["ilp720"] == "infeasible_proof").sum())

    # parameter sets ILP rules out on its own: >=1 nonabelian group, none
    # realized, every group's final ILP verdict infeasible_proof
    def _ilp_alone(d):
        return len(d) > 0 and (d["status"] != "realized").all() and (d["ilp"] == "infeasible_proof").all()
    _alone = pairs.groupby(KEY).apply(_ilp_alone, include_groups=False)
    _alone_keys = set(_alone[_alone].index)

    ilp_ruled_out = param_status[param_status["category"] == "ruled out"].copy()
    ilp_ruled_out = ilp_ruled_out.rename(columns={"infeasible": "groups_ruled_out"})
    ilp_ruled_out["ilp_alone"] = [
        tuple(r) in _alone_keys for r in ilp_ruled_out[KEY].itertuples(index=False)
    ]
    ilp_ruled_out = ilp_ruled_out[KEY + ["groups", "groups_ruled_out", "ilp_alone"]].reset_index(drop=True)
    _n_ro = len(ilp_ruled_out)
    _n_alone = int(ilp_ruled_out["ilp_alone"].sum())

    mo.vstack([
        mo.md(f"""
        | ILP nonexistence | count |
        |---|---|
        | (group, parameter) pairs proved infeasible | **{_pairs_inf:,}** |
        | &nbsp;&nbsp;in pass 1 (30 min, 10 cores) | {_p1:,} |
        | &nbsp;&nbsp;in pass 2 (12 h, 2 cores) | {_p2:,} |
        | parameter sets with no Cayley DSRG on any nonabelian group | **{_n_ro}** |
        | &nbsp;&nbsp;proved by ILP alone | {_n_alone} |
        | &nbsp;&nbsp;also relying on the legacy exhaustive search (small $n$) | {_n_ro - _n_alone} |

        With abelian groups excluded (Klin et al.), these **{_n_ro}** parameter
        sets are not realizable as a Cayley DSRG on *any* group of their order.
        The dihedral theorem (section H) closes further parameter sets beyond
        these.
        """),
        mo.md(f"**Every parameter set ruled out** ({_n_ro} rows); `groups` = "
              "nonabelian groups of order $n$, all of them eliminated; "
              "`ilp_alone` = closed without the legacy exhaustive proofs:"),
        mo.ui.table(ilp_ruled_out, selection=None, page_size=20),
    ])
    return


@app.cell
def _(mo, pairs, param_status):
    # per-order breakdown of ILP eliminations
    _pi = (
        pairs.assign(ilp_inf=(pairs["ilp"] == "infeasible_proof"))
        .groupby("n")["ilp_inf"].sum().astype(int).rename("pairs_ilp_infeasible")
    )
    _ro = (
        param_status[param_status["category"] == "ruled out"]
        .groupby("n").size().rename("params_ruled_out")
    )
    ilp_by_n = _pi.to_frame().join(_ro, how="left").fillna(0).astype(int)
    ilp_by_n = ilp_by_n[ilp_by_n["pairs_ilp_infeasible"] > 0].reset_index()
    mo.vstack([
        mo.md("ILP eliminations by order:"),
        mo.ui.table(ilp_by_n, selection=None, page_size=60),
    ])
    return


if __name__ == "__main__":
    app.run()
