#!/bin/bash
# Build the ILP pass-2 manifest: one "n k t lambda mu lib_id" line per
# (group, parameter set) pair that is neither realized nor proven infeasible.
#
#   scripts/gen_pending.sh > pending.txt
#   sbatch --array=1-"$(wc -l < pending.txt)" scripts/slurm/ilp_pass2.sh
#
# The pair universe comes from the RRHC searches CSVs. rrhc writes one row per
# swept group including the negatives, so a completed sweep enumerates every pair
# and no GAP call is needed here. A pair drops out if any run found a DPDS for it
# (rrhc at any budget, or ilp) or if ilp pass 1 proved it infeasible; what is left
# is the pairs still open, including any whose pass-1 task died before writing a
# CSV and so never reported at all.
#
# Only this run's own output is consulted, deliberately: the pass-2 scope has to
# be reproducible from the manifest alone, not a function of whatever data/ had
# accumulated from earlier campaigns.
#
# Override via env: RRHC_OUT (default rrhc-out), ILP_OUT (default ilp-out).
set -euo pipefail

shopt -s nullglob
rrhc=("${RRHC_OUT:-rrhc-out}"/r*/searches.*.csv)
ilp=("${ILP_OUT:-ilp-out}"/searches.*.csv)
[ "${#rrhc[@]}" -gt 0 ] || { echo "no rrhc searches CSVs found" >&2; exit 1; }

# key: the six columns identifying a (parameter set, group) pair.
key() { awk -F, -v want="$1" '!/^n,/ && (want == "" || $8 == want) { print $1","$2","$3","$4","$5","$6 }' "${@:2}" | sort -u; }

universe=$(key "" "${rrhc[@]}")
settled=$(
    { key found "${rrhc[@]}" ${ilp[@]+"${ilp[@]}"}
      [ "${#ilp[@]}" -gt 0 ] && key infeasible_proof "${ilp[@]}"
    } | sort -u
)

echo "# universe $(echo "$universe" | grep -c .), settled $(echo "$settled" | grep -c .)" >&2
comm -23 <(echo "$universe") <(echo "$settled") | tr ',' ' '
