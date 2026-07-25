#!/bin/bash
#SBATCH --job-name=rrhc
#SBATCH --output=rrhc-out/slurm-%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32

# One rrhc DSRG task: sweep every nonabelian group of a single parameter set and
# write the found connection sets to a dpds-schema CSV.
# Array task N reads line N of the manifest, so the two map one-to-one.
#
# Everything -- the slurm stdout log, the dpds CSV, and the searches CSV -- is
# saved under rrhc-out/. Slurm opens the --output log before the job runs, so
# that directory must already exist; create it once, then submit:
#   mkdir -p rrhc-out
#   sbatch --array=1-"$(wc -l < manifest.txt)" scripts/slurm/rrhc.sh
#
# Runs locally without slurm too -- pass the task index as $1:
#   scripts/slurm/rrhc.sh 3
#
# Override any default via env: MANIFEST (job manifest, default manifest.txt),
# RESTARTS (per-group restart budget, default 10000), OUTDIR (output dir,
# default rrhc-out), SEED (unset -> entropy), RRHC (path to the binary).

set -euo pipefail

task="${SLURM_ARRAY_TASK_ID:-${1:-}}"
: "${task:?task index required: set SLURM_ARRAY_TASK_ID or pass it as \$1}"

# Slurm copies the batch script to a spool dir, so BASH_SOURCE points there, not
# at the repo. Anchor on SLURM_SUBMIT_DIR under slurm; fall back to the script's
# real location for local runs.
repo="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
manifest="${MANIFEST:-manifest.txt}"
restarts="${RESTARTS:-10000}"
outdir="${OUTDIR:-rrhc-out}"
rrhc="${RRHC:-$repo/rrhc/target/release/rrhc}"

read -r n k t lambda mu _ < <(sed -n "${task}p" "$manifest")
[ -n "${mu:-}" ] || { echo "$manifest has no line $task (or too few columns)" >&2; exit 1; }

mkdir -p "$outdir"
dpds="$outdir/dpds.n$(printf '%03d' "$n").$task.csv"
searches="$outdir/searches.$task.csv"

seed_arg=()
[ -n "${SEED:-}" ] && seed_arg=(--seed "$SEED")

# ${arr[@]+"${arr[@]}"} expands to nothing when the array is empty, which keeps
# `set -u` happy on bash 3.2 (macOS) where a bare "${arr[@]}" would abort.
exec "$rrhc" dsrg "$n" "$k" "$t" "$lambda" "$mu" --all-groups \
    -r "$restarts" ${seed_arg[@]+"${seed_arg[@]}"} --dpds-out "$dpds" --searches-out "$searches"
