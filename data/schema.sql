-- Schema for dsrg.sqlite, dumped by scripts/export_csvs.py.
-- Rebuilt from the CSVs by scripts/import_csvs.py.

CREATE TABLE groups (
    group_id   INTEGER PRIMARY KEY,
    order_n    INTEGER NOT NULL,
    lib_id     INTEGER NOT NULL,
    name       TEXT    NOT NULL,
    is_abelian INTEGER NOT NULL,
    UNIQUE (order_n, lib_id)
);

CREATE TABLE parameters (
    param_id         INTEGER PRIMARY KEY,
    n                INTEGER NOT NULL,
    k                INTEGER NOT NULL,
    t                INTEGER NOT NULL,
    lambda           INTEGER NOT NULL,
    mu               INTEGER NOT NULL,
    existence_status TEXT    NOT NULL,
    UNIQUE (n, k, t, lambda, mu)
);

CREATE TABLE dpds (
    dpds_id       INTEGER PRIMARY KEY,
    group_id      INTEGER NOT NULL REFERENCES groups(group_id),
    param_id      INTEGER NOT NULL REFERENCES parameters(param_id),
    members       TEXT    NOT NULL,
    digraph6      TEXT,
    source_method TEXT    NOT NULL,
    graph_id INTEGER REFERENCES graphs(graph_id));

CREATE TABLE group_param (
    group_id INTEGER NOT NULL REFERENCES groups(group_id),
    param_id INTEGER NOT NULL REFERENCES parameters(param_id),
    PRIMARY KEY (group_id, param_id)
);

CREATE TABLE searches (
    search_id   INTEGER PRIMARY KEY,
    group_id    INTEGER NOT NULL REFERENCES groups(group_id),
    param_id    INTEGER NOT NULL REFERENCES parameters(param_id),
    method      TEXT    NOT NULL,   -- exhaustive | rrhc-{restarts} | ilp-{minutes}
                                    -- (or bare rrhc/ilp for pre-budget-tracking rows)
    outcome     TEXT    NOT NULL,   -- found | infeasible_proof | empty_proof
                                    -- | timeout | heuristic_none
    is_proof    INTEGER NOT NULL,   -- 1 = definitively resolves this (group,param)
                                    --     for Cayley graphs (existence or nonexistence)
    num_dpds    INTEGER NOT NULL,   -- constructions found (0 for negatives)
    num_records INTEGER NOT NULL,               -- one representative source
    UNIQUE (group_id, param_id, method)
);

CREATE INDEX idx_dpds_group ON dpds(group_id);

CREATE INDEX idx_dpds_param ON dpds(param_id);

CREATE INDEX idx_dpds_digraph6 ON dpds(digraph6);

CREATE UNIQUE INDEX uq_dpds_group_members ON dpds(group_id, members);

CREATE INDEX idx_searches_gp ON searches(group_id, param_id);

CREATE INDEX idx_searches_outcome ON searches(outcome);

CREATE VIEW v_group_param AS
SELECT g.order_n, g.lib_id, g.name AS group_name,
       p.k, p.t, p.lambda, p.mu, p.existence_status,
       g.group_id, p.param_id
FROM group_param gp
JOIN groups g     ON g.group_id = gp.group_id
JOIN parameters p ON p.param_id = gp.param_id;

CREATE INDEX idx_dpds_graph ON dpds(graph_id);

CREATE TABLE graphs (
    graph_id INTEGER PRIMARY KEY,
    digraph6 TEXT NOT NULL UNIQUE,
    param_id INTEGER NOT NULL REFERENCES parameters(param_id),
    n INTEGER NOT NULL,
    aut_order TEXT NOT NULL,
    is_drr INTEGER NOT NULL,
    is_self_converse INTEGER NOT NULL,
    num_dpds INTEGER NOT NULL,
    num_groups INTEGER NOT NULL
);

CREATE VIEW v_graphs AS
SELECT gr.graph_id, gr.digraph6, gr.n, p.k, p.t, p.lambda, p.mu, p.existence_status,
       gr.aut_order, gr.is_drr, gr.is_self_converse, gr.num_dpds, gr.num_groups
FROM graphs gr JOIN parameters p ON p.param_id=gr.param_id;

CREATE VIEW v_searches AS
SELECT s.search_id,
       g.order_n, g.lib_id, g.name AS group_name, g.is_abelian,
       p.k, p.t, p.lambda, p.mu, p.existence_status,
       s.method, s.outcome, s.is_proof, s.num_dpds, s.num_records
FROM searches s
JOIN groups g     ON g.group_id = s.group_id
JOIN parameters p ON p.param_id = s.param_id;

CREATE VIEW v_dpds AS
SELECT d.dpds_id, g.order_n, g.lib_id, g.name AS group_name,
       p.k, p.t, p.lambda, p.mu, p.existence_status,
       d.members, d.graph_id, d.digraph6,
       d.source_method
FROM dpds d JOIN groups g ON g.group_id=d.group_id
JOIN parameters p ON p.param_id=d.param_id;

