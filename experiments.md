# Setup work
- Instal environment: clone repo, uv, uv sync, ect.
- Run `scripts/gen_manifest.sh --max-n <MAX_N> > manifest.txt`
- Dispatch

# RRHC sweep

Experimental unit is a parameter set, so one array task per manifest line and one
array per restart budget. At `--max-n 110` that is 487 tasks x 7 budgets = 3409,
but each array is only 487 tasks, so none approaches femto's 1000 array cap.

Each task takes a whole node: femto is 2 sockets x 16 cores = 32 cores, and rrhc
spreads restarts across them with rayon. That caps concurrency at 30 tasks (one
per node), which is fine — they queue.

sbatch --job

```bash
mkdir -p rrhc-out/r{10,50,100,500,1000,5000,10000, 100000}

for r in 10 50 100 500 1000 5000 10000 100000; do
    sbatch --job-name="rrhc-r$r" \--array=1-"$(wc -l < manifest.txt)" \
           --output="rrhc-out/r$r/slurm-%A_%a.out" \
           --export=ALL,RESTARTS="$r",OUTDIR="rrhc-out/r$r",SEED=20260725 \
           scripts/slurm/rrhc.sh
done
```

Create the subdirectories first. Slurm opens the `--output` log before the job
starts, so the `mkdir -p` inside `rrhc.sh` happens too late for the log itself.

`rrhc.sh` defaults to `--time=12:00:00`. This is a conservative estimate and most tasks will take much less than this, especially at small `r`.

One seed across all seven budgets, deliberately. `rrhc.rs` seeds restart *i* from
`base_seed + i`, so restart *i* is the same trial at every budget: r=50 replays
r=10's ten restarts and adds forty more. The budgets nest, so "solved at r=10"
implies "solved at r=50", and the smallest budget that finds a given DPDS can be
read straight off the sweep. Drop `SEED` to seed from entropy instead, at the
cost of that property and of reproducibility.

Merging the budgets is *not* a loop over `ingest_rrhc_out.py`: it keys searches on
`(n,k,t,lambda,mu,lib_id,method)` and all seven budgets write `method=rrhc`, so
the first one ingested would mask the rest. Decide which budget is authoritative
for `data/` (or how to record the budget) before merging.

# ILP pass 1

Same shape as the RRHC sweep: experimental unit is a parameter set, one array
task per manifest line, 487 tasks at `--max-n 110`. There is no budget sweep, so
it is a single array rather than seven.

Each task sweeps every nonabelian group of its order with a 30 minute limit per
group, on 10 cores — past that the solver stops paying for the extra threads.
That is 18438 (group, parameter set) pairs in total, though the timeout is a
ceiling and most pairs settle in seconds.

```bash
mkdir -p ilp-out

sbatch --array=1-"$(wc -l < manifest.txt)" scripts/slurm/ilp_pass1.sh
```

Same directory rule as RRHC: `ilp-out` has to exist before submitting, because
slurm opens the `--output` log before `ilp_pass1.sh` gets to `mkdir -p`.

Output is the same two file kinds `rrhc` writes, to the same schema, so it
merges the same way:

```bash
scripts/ingest_rrhc_out.py ilp-out          # dry run
scripts/ingest_rrhc_out.py ilp-out --apply
```

`TIMEOUT` (default 1800s) sets the per-group limit and `QUADRATIC=1` switches
formulation; both go through `--export=ALL,...` as with RRHC.

Note that `outcome=found` covers a solve that hit the time limit but had already
produced a connection set. The model has a constant objective, so it asks a pure
feasibility question and a solution answers it — only a timeout with nothing to
show for it is recorded as `timeout`.

# ILP pass 2

Experimental unit drops to a single (parameter set, group) pair, each given 12
hours. Scope is every pair that pass 1 left neither realized nor proven
infeasible — 11575 of the 18438 pairs.

`scripts/gen_pending.sh` derives that list. It needs no GAP call: rrhc writes a
searches row for every group it sweeps including the negatives, so a completed
RRHC sweep already enumerates the whole pair universe. Subtract the pairs any run
found a DPDS for and the pairs pass 1 proved infeasible, and what remains is
still open — including pairs whose pass-1 task died before writing a CSV, which
are simply absent and so counted as open.

It reads only this run's own `rrhc-out/` and `ilp-out/`, never `data/`. The scope
has to follow from the manifest alone, not from what earlier campaigns happened to
have settled.

2 cores per task, not 10. Slurm hands out memory in proportion to cores (~2.9G
each), so 2 cores fits 16 tasks per 32-core node and 480 across femto, against 90
at 10 cores with 2 cores per node left idle. 11575 pairs / 480 concurrent = 25
waves x 12h, so under 13 days worst case.

Two scripts, worker and dispatcher: `ilp_pass2.sh` is the array task itself --
one `(parameter set, group)` pair per invocation, reading manifest line
`OFFSET + task_id`. `ilp_pass2_submit.sh` never solves anything; it regenerates
`pending.txt` and loops `sbatch --array=1-1000 --export=OFFSET=...` over it,
chunking so no single array exceeds femto's 1001-task cap. Submit the dispatcher,
not the worker, directly.

```bash
mkdir -p ilp2-out
SBATCH_ARGS="--exclude=fm10" scripts/slurm/ilp_pass2_submit.sh
```

That regenerates `pending.txt` and submits it as 12 arrays of <=1000 tasks, since
femto's `MaxArraySize` is 1001. `DRY_RUN=1` prints the `sbatch` lines instead.
`--exclude=fm10` is there because fm10 has no working DNS, which breaks Gurobi's
licence check; drop it once that is fixed.

To queue before pass 1 has drained, make it wait:

```bash
sbatch --dependency=afterany:<pass1-array-id> scripts/slurm/ilp_pass2_submit.sh
```

The dependency matters. The submitter rebuilds `pending.txt` when it runs, so
firing it after pass 1 finishes means the manifest reflects pass 1's final
output. Generating the manifest while pass-1 tasks are still running would leave
any pair they later settle in the list, and pass 2 would spend 12h re-solving it.

Merge as with everything else, but ingest pass 2 *before* pass 1:

```bash
scripts/ingest_rrhc_out.py ilp2-out --apply
scripts/ingest_rrhc_out.py ilp-out  --apply
```

Both write `method=ilp`, and ingest keys searches on
`(n,k,t,lambda,mu,lib_id,method)` keeping the first row it sees. Pass 1 recorded
`timeout` for exactly the pairs pass 2 then worked on, so ingesting pass 1 first
would let those stale timeouts mask pass 2's answers.
