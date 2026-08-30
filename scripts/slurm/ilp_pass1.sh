#!/bin/bash
#SBATCH --job-name=ilp1
#SBATCH --output=ilp-out/slurm-%A_%a.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=10

# One ILP pass-1 task: sweep every nonabelian group of a single parameter set and
# write the found connection sets to a dpds-schema CSV.
# Array task N reads line N of the manifest, so the two map one-to-one.
#
# Everything -- the slurm stdout log, the dpds CSV, and the searches CSV -- is
# saved under ilp-out/. Slurm opens the --output log before the job runs, so
# that directory must already exist; create it once, then submit:
#   mkdir -p ilp-out
#   sbatch --array=1-"$(wc -l < manifest.txt)" scripts/slurm/ilp_pass1.sh
#
# Runs locally without slurm too -- pass the task index as $1:
#   scripts/slurm/ilp_pass1.sh 3
#
# Override any default via env: MANIFEST (job manifest, default manifest.txt),
# TIMEOUT (per-pair solve limit in seconds, default 1800), OUTDIR (output dir,
# default ilp-out), QUADRATIC (set -> quadratic formulation), PYTHON (path to
# the interpreter).

set -euo pipefail

task="${SLURM_ARRAY_TASK_ID:-${1:-}}"
: "${task:?task index required: set SLURM_ARRAY_TASK_ID or pass it as \$1}"


repo="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
manifest="${MANIFEST:-manifest.txt}"
timeout="${TIMEOUT:-1800}"
outdir="${OUTDIR:-ilp-out}"
python="${PYTHON:-$repo/.venv/bin/python}"

read -r n k t lambda mu _ < <(sed -n "${task}p" "$manifest")
[ -n "${mu:-}" ] || { echo "$manifest has no line $task (or too few columns)" >&2; exit 1; }

mkdir -p "$outdir"
dpds="$outdir/dpds.n$(printf '%03d' "$n").$task.csv"
searches="$outdir/searches.$task.csv"

quad_arg=()
[ -n "${QUADRATIC:-}" ] && quad_arg=(--quadratic)

# ${arr[@]+"${arr[@]}"} expands to nothing when the array is empty, which keeps
# `set -u` happy on bash 3.2 (macOS) where a bare "${arr[@]}" would abort.
exec "$python" -m ilp search --n "$n" --k "$k" --t "$t" --lambda "$lambda" --mu "$mu" \
    --time-limit "$timeout" --threads "${SLURM_CPUS_PER_TASK:-10}" --quiet \
    ${quad_arg[@]+"${quad_arg[@]}"} --dpds-out "$dpds" --searches-out "$searches"
