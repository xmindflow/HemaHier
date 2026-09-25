#!/usr/bin/env bash
# Train Flat or HemaHier on MLLv1, five folds.
#
#   export MLLV1_ROOT=/path/to/Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer_v1
#   GPUS=0,1,2 ./scripts/run.sh hemahier
#   ./scripts/run.sh hemahier --status
#   DRY_RUN=1 ./scripts/run.sh hemahier
#
# Matrices:  flat | hemahier | flat_lora | hemahier_lora | all
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/lib/common.sh"
hemahier_setup_env
[[ -f "$REPO/scripts/local.env" ]] && source "$REPO/scripts/local.env"
PYTHON="${PYTHON:-python3}"
CMD="${1:-}"
shift || true

usage() { sed -n '2,9p' "$0"; }

case "$CMD" in
  ""|-h|--help|help) usage; exit 0 ;;
  splits) exec "$PYTHON" "$REPO/scripts/make_cv_splits.py" "$@" ;;
  download-dinobloom) exec bash "$REPO/scripts/download_dinobloom.sh" "$@" ;;
esac

STATUS_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --status) STATUS_ONLY=1; shift ;;
    *) break ;;
  esac
done

if [[ -z "${MLLV1_ROOT:-}" || ! -d "${MLLV1_ROOT}" ]]; then
  echo "ERROR: set MLLV1_ROOT to the MLLv1 image root." >&2
  exit 1
fi

FOLDS=(0 1 2 3 4)
EPOCHS=10
WARMUP=1
SEED=2026
NUM_WORKERS="${NUM_WORKERS:-8}"
DECODE_ARGS=(--decode_mode argmax)
LORA_ARGS=()
EXP=""

case "$CMD" in
  flat)
    EXP=Flat
    OUT="${OUT:-$REPO/runs/mllv1_flat}"
    ;;
  flat_lora)
    EXP=Flat
    OUT="${OUT:-$REPO/runs/mllv1_flat_lora}"
    LORA_ARGS=(--lora_rank 16 --lora_alpha 128)
    ;;
  hemahier)
    EXP=HemaHier
    OUT="${OUT:-$REPO/runs/mllv1_hemahier}"
    DECODE_ARGS=()
    ;;
  hemahier_lora)
    EXP=HemaHier_LoRA
    OUT="${OUT:-$REPO/runs/mllv1_hemahier_lora}"
    DECODE_ARGS=()
    LORA_ARGS=(--lora_rank 16 --lora_alpha 128)
    ;;
  all)
    SUB=(); [[ "$STATUS_ONLY" == "1" ]] && SUB=(--status)
    for sub in flat hemahier flat_lora hemahier_lora; do
      echo "=== $sub ==="
      "$0" "$sub" "${SUB[@]+"${SUB[@]}"}"
    done
    exit 0
    ;;
  *) echo "Unknown matrix: $CMD" >&2; usage >&2; exit 1 ;;
esac

GPUS_RAW="${GPUS:-0}"
read -r -a GPU_LIST <<< "${GPUS_RAW//,/ }"
N_GPUS="${#GPU_LIST[@]}"
LOG_DIR="$OUT/logs"

COMMON=(
  --exp_id "$EXP"
  --mllv1_root "$MLLV1_ROOT"
  --seed "$SEED" --backbone dinobloom_s --freeze_backbone
  --lr 3e-4 --batch_size 64 --num_workers "$NUM_WORKERS"
  --epochs "$EPOCHS" --warmup_epochs "$WARMUP"
  --lr_schedule cosine_epoch --loss_schedule finalshot
  "${DECODE_ARGS[@]+"${DECODE_ARGS[@]}"}"
  "${LORA_ARGS[@]+"${LORA_ARGS[@]}"}"
)

fold_complete() {
  local d="$1"
  [[ -f "$d/completed.json" && -f "$d/best_checkpoint.pt" ]]
}

if [[ "$STATUS_ONLY" == "1" ]]; then
  done_n=0
  for fold in "${FOLDS[@]}"; do
    fold_complete "$OUT/fold${fold}/$EXP" && done_n=$((done_n + 1)) || true
  done
  echo "=== $CMD  OUT=$OUT  $done_n/${#FOLDS[@]} folds complete ==="
  exit 0
fi

mkdir -p "$LOG_DIR"
echo "=== $CMD  exp=$EXP  OUT=$OUT  GPUs=${GPU_LIST[*]} ==="

run_fold() {
  local gpu="$1" fold="$2"
  local run_out="$OUT/fold${fold}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "$PYTHON $REPO/src/train.py --output_dir $run_out --fold $fold ${COMMON[*]}"
    return 0
  fi
  if [[ "${FORCE:-0}" != "1" ]] && fold_complete "$run_out/$EXP"; then
    echo "[skip] fold$fold"
    return 0
  fi
  mkdir -p "$run_out"
  echo "[start] GPU$gpu fold$fold"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO/src" "$PYTHON" "$REPO/src/train.py" \
    --output_dir "$run_out" --fold "$fold" "${COMMON[@]}" \
    >"$LOG_DIR/fold${fold}.log" 2>&1
  echo "[done] GPU$gpu fold$fold"
}

running=0
for i in "${!FOLDS[@]}"; do
  while (( running >= N_GPUS )); do
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  done
  run_fold "${GPU_LIST[$((i % N_GPUS))]}" "${FOLDS[$i]}" &
  running=$((running + 1))
done
wait
echo "Finished $CMD"
