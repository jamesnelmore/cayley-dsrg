#!/bin/bash
# Build a slurm job manifest from a parameter CSV: emit one line per open row, as
# space-separated "n k t lambda mu" (no header) so that array task N maps to
# manifest line N and the columns drop straight into `rrhc dsrg` or the ILP jobs.
#
#   scripts/gen_manifest.sh --max-n 110 > manifest.txt
#   sbatch --array=1-"$(wc -l < manifest.txt)" scripts/slurm/rrhc.sh
#
# Orders with no nonabelian group are dropped. The DSRG searches sweep only
# nonabelian groups (`rrhc --all-groups`, and ilp's list_groups), so a parameter
# set on such an order has an empty group list -- it would burn an array task to
# do nothing. GAP decides this rather than a hardcoded list, so it stays correct
# for whatever range you ask for.
#
# Columns are located by header name, so reordering the CSV cannot silently
# shift which field is read. Options:
#
#   --status S               row status to keep   (default: open)
#   --max-n N                drop orders above N  (default: no limit)
#   --keep-abelian-orders    skip the GAP filter
#
# Override the GAP binary with GAP_BIN.
set -euo pipefail

params="data/parameters.csv"
status="open"
max_n=""
gap_filter=1

while [ $# -gt 0 ]; do
    case "$1" in
        --status)  status="$2"; shift 2 ;;
        --max-n)   max_n="$2"; shift 2 ;;
        --keep-abelian-orders) gap_filter=""; shift ;;
        -h|--help) sed -n '2,22p' "$0" >&2; exit 0 ;;
        -*)        echo "unknown option: $1" >&2; exit 2 ;;
        *)         params="$1"; shift ;;
    esac
done

[ -f "$params" ] || { echo "no such parameter CSV: $params" >&2; exit 1; }

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

# sub(/\r$/,"") strips a trailing CR so CRLF-terminated CSVs still match.
# The status column is named `status` in data/parameters.csv but
# `existence_status` in the sqlite schema; accept either.
awk -F, -v want="$status" -v maxn="$max_n" '
    { sub(/\r$/, "") }
    NR == 1 {
        for (i = 1; i <= NF; i++) col[$i] = i
        split("n k t lambda mu", need, " ")
        for (f = 1; f <= 5; f++)
            if (!(need[f] in col)) { print "missing column: " need[f] > "/dev/stderr"; exit 1 }
        s = ("status" in col) ? col["status"] : col["existence_status"]
        if (!s) { print "missing column: status" > "/dev/stderr"; exit 1 }
        next
    }
    $s == want && (maxn == "" || $col["n"] + 0 <= maxn + 0) {
        print $col["n"], $col["k"], $col["t"], $col["lambda"], $col["mu"]
    }
' "$params" > "$tmp"

rows=$(wc -l < "$tmp")
[ "$rows" -gt 0 ] || { echo "no rows with status=$status${max_n:+ and n<=$max_n}" >&2; exit 1; }

if [ -z "$gap_filter" ]; then
    cat "$tmp"
    echo "# $rows parameter sets (abelian-order filter skipped)" >&2
    exit 0
fi

orders=$(cut -d' ' -f1 "$tmp" | sort -un)

# One GAP call for every distinct order. ForAny short-circuits on the first
# nonabelian group, so an order that has one costs a single group construction;
# an order that does not (n = p, p^2, or pq with q not congruent to 1 mod p) has
# only a handful of groups to exhaust. Either way this stays cheap as n grows.
# `tr -d '\r'`: gap -q emits a bare CR ahead of its first line of output, which
# would otherwise make that line's first field "\rKEEP" and silently drop the
# lowest order from every manifest.
gap_out=$("${GAP_BIN:-gap}" -q <<GAPEOF | tr -d '\r' || true
LoadPackage("smallgrp");;
for n in [$(echo "$orders" | paste -sd,)] do
    if not SmallGroupsAvailable(n) then
        Print("UNAVAILABLE ", n, "\n");
    elif ForAny([1..NrSmallGroups(n)], i -> not IsAbelian(SmallGroup(n, i))) then
        Print("KEEP ", n, "\n");
    else
        Print("DROP ", n, "\n");
    fi;
od;
Print("DONE\n");
QUIT;
GAPEOF
)

# Fail loudly rather than emit a manifest built on a partial group survey.
echo "$gap_out" | grep -qx "DONE" || {
    echo "GAP did not finish; refusing to emit a manifest. Output was:" >&2
    echo "$gap_out" >&2
    exit 1
}
unavailable=$(echo "$gap_out" | awk '$1=="UNAVAILABLE"{print $2}' | paste -sd' ')
if [ -n "$unavailable" ]; then
    echo "SmallGroups library has no data for order(s): $unavailable" >&2
    exit 1
fi

keep=$(echo "$gap_out" | awk '$1=="KEEP"{print $2}')
dropped=$(echo "$gap_out" | awk '$1=="DROP"{print $2}' | paste -sd' ')

kept="$(mktemp)"
trap 'rm -f "$tmp" "$kept"' EXIT

awk -v keep="$keep" '
    BEGIN { n = split(keep, a, "\n"); for (i = 1; i <= n; i++) ok[a[i] + 0] = 1 }
    ok[$1 + 0]
' "$tmp" > "$kept"

kept_rows=$(wc -l < "$kept")
# An empty manifest is never what the caller wants: `--array=1-0` is a submit
# error, so say so here rather than let sbatch fail further downstream.
[ "$kept_rows" -gt 0 ] || {
    echo "every matching row is on an abelian-only order: $dropped" >&2
    echo "nothing to search (the DSRG sweeps skip abelian groups)" >&2
    exit 1
}

cat "$kept"

echo "# $kept_rows parameter sets over $(echo "$keep" | grep -c .) orders" >&2
[ -n "$dropped" ] && echo "# dropped $((rows - kept_rows)) row(s) on abelian-only order(s): $dropped" >&2
exit 0
