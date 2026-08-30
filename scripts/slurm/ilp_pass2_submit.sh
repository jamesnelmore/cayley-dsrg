#!/bin/bash
#SBATCH --job-name=ilp2-submit
#SBATCH --output=ilp2-out/submit-%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1

# Regenerate the pass-2 manifest and submit it as chunked arrays.
#
# Run it directly once pass 1 has finished:
#   mkdir -p ilp2-out
#   scripts/slurm/ilp_pass2_submit.sh
#
# Or queue it now to fire the moment pass 1 drains, so the manifest is built from
# pass 1's *final* output rather than a snapshot taken while tasks are still
# running:
#   sbatch --dependency=afterany:<pass1-array-id> scripts/slurm/ilp_pass2_submit.sh
#
# Regenerating here rather than reusing a pending.txt from earlier is the point: a
# pass-1 task that finishes after the manifest was written would leave its settled
# pairs in the list and get them re-solved for 12h apiece.
#
# Override via env: MANIFEST (default pending.txt), CHUNK (tasks per array,
# default 1000 -- femto's MaxArraySize is 1001), SBATCH_ARGS (extra args, e.g.
# "--exclude=fm10"), DRY_RUN (set -> print the sbatch lines without submitting).

set -euo pipefail

repo="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$repo"

manifest="${MANIFEST:-pending.txt}"
chunk="${CHUNK:-1000}"

"$repo/scripts/gen_pending.sh" > "$manifest"
total=$(wc -l < "$manifest")
[ "$total" -gt 0 ] || { echo "$manifest is empty; nothing to submit" >&2; exit 1; }
echo "# $manifest: $total pairs -> $(( (total + chunk - 1) / chunk )) array job(s)" >&2

mkdir -p ilp2-out
# shellcheck disable=SC2086  # SBATCH_ARGS is deliberately word-split
for (( off = 0; off < total; off += chunk )); do
    size=$(( total - off )); [ "$size" -gt "$chunk" ] && size="$chunk"
    if [ -n "${DRY_RUN:-}" ]; then
        echo "sbatch ${SBATCH_ARGS:-} --array=1-$size --export=ALL,OFFSET=$off scripts/slurm/ilp_pass2.sh"
    else
        sbatch ${SBATCH_ARGS:-} --array=1-"$size" --export=ALL,OFFSET="$off" \
            "$repo/scripts/slurm/ilp_pass2.sh"
    fi
done
