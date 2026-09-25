#!/usr/bin/env bash
# Download the released MLLv1 checkpoints and check them against SHA256SUMS.
#
#   ./scripts/fetch_weights.sh              # both settings
#   ./scripts/fetch_weights.sh hemahier     # frozen backbone only
#   ./scripts/fetch_weights.sh hemahier_lora
#
# Override the source with WEIGHTS_BASE_URL=... (e.g. a local mirror).
# Files land in weights/mllv1/, which is gitignored.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${WEIGHTS_TAG:-mllv1}"
BASE="${WEIGHTS_BASE_URL:-https://github.com/xmindflow/HemaHier/releases/download/$TAG}"
DEST="$REPO/weights/mllv1"

# sha256 of each tar, as released.
declare -A SUMS=(
  [mllv1_hemahier.tar]=7b0936c5d2622c6e2046da7b5b3a789659a26325867882f7cfbdf1d78b6b839b
  [mllv1_hemahier_lora.tar]=fec7be289179f0b67be9a6cc38f5be71c3984c802594c9b69d687585faa25036
)

case "${1:-all}" in
  all)            assets=(mllv1_hemahier.tar mllv1_hemahier_lora.tar) ;;
  hemahier)       assets=(mllv1_hemahier.tar) ;;
  hemahier_lora)  assets=(mllv1_hemahier_lora.tar) ;;
  -h|--help)      sed -n '2,10p' "$0"; exit 0 ;;
  *) echo "Unknown target: $1 (use: all | hemahier | hemahier_lora)" >&2; exit 1 ;;
esac

mkdir -p "$DEST"
for asset in "${assets[@]}"; do
  tarball="$DEST/$asset"
  if [[ ! -f "$tarball" ]]; then
    echo ">>> downloading $asset"
    curl -fL --progress-bar -o "$tarball.part" "$BASE/$asset"
    mv "$tarball.part" "$tarball"
  fi
  echo ">>> verifying $asset"
  echo "${SUMS[$asset]}  $tarball" | sha256sum -c -
  tar -xf "$tarball" -C "$DEST"
done

echo
echo "Checkpoints in $DEST. Re-score one fold with:"
echo "  PYTHONPATH=src python src/evaluate.py \\"
echo "      $DEST/mllv1_hemahier/fold0/HemaHier_Cascade_Probe --dataset_root \"\$MLLV1_ROOT\""
