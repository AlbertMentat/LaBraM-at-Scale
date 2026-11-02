#!/usr/bin/env bash
# Compatibility entrypoint; use 25k_base.sh or 25k_large.sh for an explicit size.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
case "${MODEL:-labram_large_patch200_1600_8k_vocab}" in
  labram_base_patch200_1600_8k_vocab) bash "$HERE/25k_base.sh" "$@" ;;
  labram_large_patch200_1600_8k_vocab) bash "$HERE/25k_large.sh" "$@" ;;
  *) echo 'Use a size-specific Quark recipe for this model.' >&2; exit 2 ;;
esac
