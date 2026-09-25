#!/usr/bin/env bash
# Download DinoBloom checkpoints (default: S, the backbone used in the paper).
#
# Usage:
#   ./scripts/download_dinobloom.sh           # dinobloom_s
#   ./scripts/download_dinobloom.sh s b       # specific variants (s|b|l|g)
#   VARIANTS="s" ./scripts/download_dinobloom.sh

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/lib/common.sh"
hemahier_setup_env

VARIANTS="${VARIANTS:-s}"
if [[ $# -gt 0 ]]; then
  VARIANTS="$*"
fi

map_variant() {
  case "$1" in
    s) echo "dinobloom_s" ;;
    b) echo "dinobloom_b" ;;
    l) echo "dinobloom_l" ;;
    g) echo "dinobloom_g" ;;
    dinobloom_*) echo "$1" ;;
    *) echo "Unknown variant: $1" >&2; exit 1 ;;
  esac
}

echo "DinoBloom weights -> $REPO/weights/dinobloom/"
for v in $VARIANTS; do
  name="$(map_variant "$v")"
  echo ">>> $name"
  PYTHONPATH="$REPO/src" python3 -c "
from models.dinobloom import ensure_dinobloom_weights
p = ensure_dinobloom_weights('$name')
print('OK:', p)
"
done
echo "Done."
