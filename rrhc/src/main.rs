mod group;
mod rrhc;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::{ArgGroup, Args, Parser, Subcommand};
use rayon::prelude::*;

use group::Group;
use rrhc::{
    DSRGParameterSet, SRGParameterSet, ScoredSet, random_restart_hill_climb,
    random_restart_hill_climb_srg,
};

/// Search a group's Cayley table for a (directed) strongly regular Cayley graph
/// -- a connection set solving the (D)SRG equation -- by parallel random-restart
/// hill climbing. Prints `FOUND <element indices>` or `NONE` per group.
///
/// Two search modes:
///   dsrg  directed strongly regular graphs (Brouwer/Hobart), 5 parameters,
///         swept over the nonabelian groups of the order;
///   srg   strongly regular graphs (Brouwer's SRG tables) via inverse-closed
///         partial difference sets, 4 parameters, swept over *all* groups
///         (abelian included).
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Directed strongly regular graph: connection set of size k with
    /// parameters (t, lambda, mu) in a group of order n.
    Dsrg(DsrgArgs),
    /// Strongly regular graph via an inverse-closed partial difference set:
    /// connection set of size k with parameters (lambda, mu) in a group of
    /// order v. Equivalent to a DSRG with t = k, but the set is inverse-closed
    /// (its Cayley graph is undirected), searched by moving over inverse-pairs.
    Srg(SrgArgs),
}

#[derive(Args)]
struct DsrgArgs {
    /// Group order
    n: usize,
    /// Connection set size
    k: usize,
    /// DSRG parameter t
    t: usize,
    /// DSRG parameter lambda
    lambda: usize,
    /// DSRG parameter mu
    mu: usize,
    /// Write the found connection sets to a dpds-schema CSV
    /// (`n,k,t,lambda,mu,lib_id,members,source_method`; `members` 1-based per
    /// data/schema.md). One invocation is a single order n, so the file drops
    /// straight into data/dpds/nNNN.csv.
    #[arg(long, value_name = "PATH")]
    dpds_out: Option<PathBuf>,
    /// Write one searches-schema row per swept group -- including the negatives
    /// -- to a CSV (`n,k,t,lambda,mu,lib_id,method,outcome,num_dpds`).
    #[arg(long, value_name = "PATH")]
    searches_out: Option<PathBuf>,
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Args)]
struct SrgArgs {
    /// Group order (number of vertices v)
    v: usize,
    /// Connection set size (valency k)
    k: usize,
    /// SRG parameter lambda
    lambda: usize,
    /// SRG parameter mu
    mu: usize,
    #[command(flatten)]
    common: CommonArgs,
}

/// Options shared by both search modes: which group(s) to search plus the
/// restart budget and seed. Either give a specific `group_id`, or pass
/// `--all-groups` to sweep the whole order.
#[derive(Args)]
#[command(group(ArgGroup::new("target").required(true).args(["group_id", "all_groups"])))]
struct CommonArgs {
    /// SmallGroup library id (the i in SmallGroup(order, i))
    group_id: Option<u64>,
    /// Sweep every candidate group of the order instead of a single id
    /// (dsrg: nonabelian groups only; srg: all groups)
    #[arg(long)]
    all_groups: bool,
    /// Maximum number of hill-climbing restarts
    #[arg(short = 'r', long, default_value_t = 1000)]
    max_restarts: usize,
    /// Seed for reproducible runs; each group's RNG stream is derived from it.
    /// Omit to seed from entropy (results vary run to run).
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    if let Err(err) = run(Cli::parse()) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn run(cli: Cli) -> Result<(), BoxError> {
    match cli.command {
        Command::Dsrg(args) => {
            let params = DSRGParameterSet {
                n: args.n,
                k: args.k,
                t: args.t,
                lambda: args.lambda,
                mu: args.mu,
            };
            let restarts = args.common.max_restarts;
            // DSRGs are searched on the nonabelian groups only.
            let results = execute(args.n, &args.common, false, move |group, seed, progress| {
                random_restart_hill_climb(group, &params, restarts, seed, progress)
            })?;
            if let Some(path) = args.dpds_out.as_deref() {
                write_dpds(path, args.n, args.k, args.t, args.lambda, args.mu, restarts, &results)?;
            }
            if let Some(path) = args.searches_out.as_deref() {
                write_searches(path, args.n, args.k, args.t, args.lambda, args.mu, restarts, &results)?;
            }
            Ok(())
        }
        Command::Srg(args) => {
            let params = SRGParameterSet {
                v: args.v,
                k: args.k,
                lambda: args.lambda,
                mu: args.mu,
            };
            let restarts = args.common.max_restarts;
            // SRGs (inverse-closed) arise on abelian groups too: search all.
            // No CSV output for the srg mode: it keeps its streamed stdout output.
            execute(args.v, &args.common, true, move |group, seed, progress| {
                random_restart_hill_climb_srg(group, &params, restarts, seed, progress)
            })?;
            Ok(())
        }
    }
}

/// Load the target group(s), run `search` on each in parallel, and stream one
/// status line to stdout as each group finishes. Returns the results sorted by
/// id so the caller can write whatever CSV artifacts it wants.
///
/// `all_groups_include_abelian` selects the sweep source when `--all-groups` is
/// given: every group of the order (SRG) versus only the nonabelian ones (DSRG).
/// A specific `group_id` is always loaded directly, abelian or not.
fn execute<F>(
    order: usize,
    common: &CommonArgs,
    all_groups_include_abelian: bool,
    search: F,
) -> Result<Vec<(u64, ScoredSet)>, BoxError>
where
    F: Fn(&Group, u64, &AtomicUsize) -> ScoredSet + Sync,
{
    if common.max_restarts == 0 {
        return Err("max_restarts must be at least 1".into());
    }

    // Load all the Cayley tables from a single GAP process up front (one GAP
    // startup, not one per group), then hand the queue to the search below.
    // Exactly one of group_id / --all-groups is set (enforced by clap).
    println!("loading groups of order {order} from GAP...");
    let groups: Vec<(u64, Group)> = match common.group_id {
        Some(id) => vec![(id, Group::from_gap(order as u64, id)?)],
        None if all_groups_include_abelian => Group::load_all(order as u64)?,
        None => Group::load_nonabelian(order as u64)?,
    };
    if groups.is_empty() {
        println!("no candidate groups of order {order}");
        return Ok(Vec::new());
    }

    let total = groups.len();
    println!(
        "loaded {total} group(s); searching up to {} restarts each on {} core(s)...",
        common.max_restarts,
        rayon::current_num_threads()
    );
    // Required by the search API for per-restart accounting; not displayed.
    let restarts_done = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);

    // Workers pull groups off the queue in parallel and search each; the inner
    // restart search is parallel too (rayon shares one pool).
    let mut results: Vec<(u64, ScoredSet)> = groups
        .par_iter()
        .map(|(id, group)| {
            // Derive a well-separated per-group seed so a fixed --seed
            // reproduces regardless of core count and completion order.
            let base_seed = match common.seed {
                Some(seed) => seed.wrapping_add(id.wrapping_mul(0x9E3779B97F4A7C15)),
                None => rand::random::<u64>(),
            };
            let best = search(group, base_seed, &restarts_done);
            // One status line per group as it finishes, in completion order.
            // `println!` locks stdout, so parallel workers never interleave.
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            println!("[{done}/{total}] id={id}: {}", summarize(&best));
            (*id, best)
        })
        .collect();

    // Sort by id so the CSV artifacts the caller writes are deterministic.
    results.sort_by_key(|(id, _)| *id);
    let found = results.iter().filter(|(_, best)| best.error == 0).count();
    println!(
        "swept {total} group(s): {found} found, {} none",
        total - found
    );

    Ok(results)
}

/// Write the FOUND connection sets as dpds-schema CSV rows
/// (`n,k,t,lambda,mu,lib_id,members,source_method`), one per group that solved
/// the equation. The parameters are the same for every row (one invocation is
/// one parameter set). `members` is 1-based (per data/schema.md's `Elements`-order
/// encoding); `results` is assumed already sorted by id.
///
/// `source_method` carries the restart budget (`rrhc-{restarts}`), same as
/// `write_searches` -- a bare `rrhc` can't tell a fresh run's construction apart
/// from an older one at a different budget, which is exactly what let one
/// ingest silently erase the other.
#[allow(clippy::too_many_arguments)]
fn write_dpds(
    path: &Path,
    n: usize,
    k: usize,
    t: usize,
    lambda: usize,
    mu: usize,
    restarts: usize,
    results: &[(u64, ScoredSet)],
) -> Result<(), BoxError> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "n,k,t,lambda,mu,lib_id,members,source_method")?;
    for (id, best) in results {
        if best.error == 0 {
            let members: Vec<String> = best
                .connection_set
                .iter()
                .map(|e| (e + 1).to_string())
                .collect();
            writeln!(
                w,
                "{n},{k},{t},{lambda},{mu},{id},{},rrhc-{restarts}",
                members.join(" ")
            )?;
        }
    }
    w.flush()?;
    Ok(())
}

/// Write one searches-schema row per swept group -- including the negatives, the
/// record that can't be recovered from the constructions -- as CSV
/// (`n,k,t,lambda,mu,lib_id,method,outcome,num_dpds`). The parameters are the
/// same for every row (one invocation is one parameter set); `results` is
/// assumed already sorted by id.
///
/// `method` encodes the restart budget as `rrhc-{restarts}` (e.g. `rrhc-10000`)
/// rather than a bare `rrhc`, since budget is part of what identifies a search:
/// two budgets on the same (group, parameter set) are different experiments,
/// not a repeat of the same one, and searches.csv keys rows on `method` -- this
/// is what lets both survive ingestion instead of one masking the other.
#[allow(clippy::too_many_arguments)]
fn write_searches(
    path: &Path,
    n: usize,
    k: usize,
    t: usize,
    lambda: usize,
    mu: usize,
    restarts: usize,
    results: &[(u64, ScoredSet)],
) -> Result<(), BoxError> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "n,k,t,lambda,mu,lib_id,method,outcome,num_dpds")?;
    for (id, best) in results {
        // rrhc dsrg only ever finds (error 0) or fails to (a positive best
        // error); it never proves nonexistence, so a miss is `heuristic_none`.
        let (outcome, num_dpds) = if best.error == 0 {
            ("found", 1)
        } else {
            ("heuristic_none", 0)
        };
        writeln!(
            w,
            "{n},{k},{t},{lambda},{mu},{id},rrhc-{restarts},{outcome},{num_dpds}"
        )?;
    }
    w.flush()?;
    Ok(())
}

/// One-line summary of a result: `FOUND <indices>` (1-based, per data/schema.md),
/// `NONE (best error N)`, or, for the SRG search, `INFEASIBLE` when the group has
/// no inverse-closed set of size k at all (signalled by `error == i64::MAX`).
fn summarize(best: &ScoredSet) -> String {
    if best.error == 0 {
        let elements: Vec<String> = best
            .connection_set
            .iter()
            .map(|e| (e + 1).to_string())
            .collect();
        format!("FOUND {}", elements.join(" "))
    } else if best.error == i64::MAX {
        "INFEASIBLE (no inverse-closed set of size k in this group)".to_string()
    } else {
        format!("NONE (best error {})", best.error)
    }
}
