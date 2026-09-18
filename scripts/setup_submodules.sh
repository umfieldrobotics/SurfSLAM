#!/usr/bin/env bash
# Initialize the model submodules and apply SurfSLAM's patches to them.
#
# Submodules are pinned to exact upstream commits; our changes live in
# patches/<name>.patch and are applied to the submodule working tree here.
# Idempotent: safe to re-run at any time.
#
# Usage:
#   ./scripts/setup_submodules.sh          # init submodules and apply patches
#   ./scripts/setup_submodules.sh --check  # report state, change nothing
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK=0
if [[ "${1:-}" == "--check" ]]; then
    CHECK=1
elif [[ -n "${1:-}" ]]; then
    echo "usage: $0 [--check]" >&2
    exit 2
fi

cd "$ROOT"

if [[ $CHECK -eq 0 ]]; then
    git submodule update --init
fi

status=0
for patch in "$ROOT"/patches/*.patch; do
    name="$(basename "$patch" .patch)"
    sub="$ROOT/submodules/$name"
    if [[ ! -e "$sub/.git" ]]; then
        echo "[missing ] $name: submodule not initialized"
        status=1
    elif git -C "$sub" apply --reverse --check "$patch" 2>/dev/null; then
        echo "[ok      ] $name: patch applied"
    elif git -C "$sub" apply --check "$patch" 2>/dev/null; then
        if [[ $CHECK -eq 1 ]]; then
            echo "[pending ] $name: patch not applied"
            status=1
        else
            git -C "$sub" apply "$patch"
            echo "[ok      ] $name: patch applied now"
        fi
    else
        echo "[conflict] $name: patch does not apply; the submodule has other local changes" >&2
        status=1
    fi
done
exit $status
