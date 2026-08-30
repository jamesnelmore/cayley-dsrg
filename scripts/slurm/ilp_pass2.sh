#!/bin/bash
#SBATCH --job-name=ilp2
#SBATCH --output=ilp2-out/slurm-%A_%a.out
#SBATCH --time=12:30:00
#SBATCH --cpus-per-task=2

# One ILP pass-2 task: solve a single (parameter set, group) pair -- the pairs
# pass 1 left neither realized nor proven infeasible -- and write the result to a
# dpds-schema CSV.
# Array task N reads line OFFSET+N of the manifest, so the two map one-to-one.
#
# OFFSET exists because femto's MaxArraySize is 1001, so an array can hold at
# most 1000 tasks while the pending list is far longer. Submit in chunks:
#   mkdir -p ilp2-out
#   scripts/gen_pending.sh > pending.txt
#   sbatch --array=1-1000 --export=ALL,OFFSET=0    scripts/slurm/ilp_pass2.sh
#   sbatch --array=1-1000 --export=ALL,OFFSET=1000 scripts/slurm/ilp_pass2.sh
#   ...
#
# Runs locally without slurm too -- pass the task index as $1:
#   scripts/slurm/ilp_pass2.sh 3
#
# 2 cores per task is deliberate: slurm hands out memory in proportion to cores
# (~2.9G each here), so 2 cores fits 16 tasks on a 32-core node and 480 across
# femto. At 10 cores only 3 fit and 2 cores per node sit idle, which costs more in
# lost concurrency than the extra threads win back on any one solve.
#
# Override any default via env: MANIFEST (job manifest, default pending.txt),
# OFFSET (manifest line offset, default 0), TIMEOUT (solve limit in seconds,
# default 43200 = 12h, leaving the 12h30m walltime room to record the result),
# OUTDIR (output dir, default ilp2-out), QUADRATIC (set -> quadratic
# formulation), PYTHON (path to the interpreter).

set -euo pipefail

task="${SLURM_ARRAY_TASK_ID:-${1:-}}"
: "${task:?task index required: set SLURM_ARRAY_TASK_ID or pass it as \$1}"
line=$(( task + ${OFFSET:-0} ))

repo="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
manifest="${MANIFEST:-pending.txt}"
timeout="${TIMEOUT:-43200}"
outdir="${OUTDIR:-ilp2-out}"
python="${PYTHON:-$repo/.venv/bin/python}"

read -r n k t lambda mu lib_id _ < <(sed -n "${line}p" "$manifest")
[ -n "${lib_id:-}" ] || { echo "$manifest has no line $line (or too few columns)" >&2; exit 1; }

mkdir -p "$outdir"
dpds="$outdir/dpds.n$(printf '%03d' "$n").$line.csv"
searches="$outdir/searches.$line.csv"

quad_arg=()
[ -n "${QUADRATIC:-}" ] && quad_arg=(--quadratic)

# ${arr[@]+"${arr[@]}"} expands to nothing when the array is empty, which keeps
# `set -u` happy on bash 3.2 (macOS) where a bare "${arr[@]}" would abort.
exec "$python" -m ilp solve --n "$n" --k "$k" --t "$t" --lambda "$lambda" --mu "$mu" \
    --lib-id "$lib_id" \
    --time-limit "$timeout" --threads "${SLURM_CPUS_PER_TASK:-10}" --quiet \
    ${quad_arg[@]+"${quad_arg[@]}"} --dpds-out "$dpds" --searches-out "$searches"
