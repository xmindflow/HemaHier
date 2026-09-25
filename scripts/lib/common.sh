#!/usr/bin/env bash
# Shared setup for HemaHier shell runners.
# Usage:  source "$REPO/scripts/lib/common.sh" && hemahier_setup_env

_HEMAHIER_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

hemahier_repo_root() {
  cd "$_HEMAHIER_LIB_DIR/../.." && pwd
}

hemahier_setup_env() {
  REPO="${REPO:-$(hemahier_repo_root)}"
  cd "$REPO"
  export PYTHONPATH="${REPO}/src"
  export CUDA_VISIBLE_DEVICES="${GPU:-0}"
}

resolve_dataset_root() {
  local var_name="$1"
  shift
  if [[ -n "${!var_name:-}" && -d "${!var_name}" ]]; then
    return 0
  fi
  for _c in "$@"; do
    if [[ -d "$_c" ]]; then
      export "$var_name=$_c"
      return 0
    fi
  done
  echo "ERROR: set $var_name or place data at one of: $*" >&2
  return 1
}

hemahier_trainers_running() {
  ps -eo comm=,args= | awk '
    $1 ~ /^python/ && $0 ~ /src\/train\.py/ { found=1 }
    END { exit(found ? 0 : 1) }
  '
}

hemahier_wait_for_idle_trainers() {
  if [[ "${ALLOW_GPU_SHARING:-0}" == "1" ]]; then
    return 0
  fi
  while hemahier_trainers_running; do
    echo "[$(date -Iseconds)] waiting for active training processes (set ALLOW_GPU_SHARING=1 to override)"
    sleep "${TRAINER_POLL_SECONDS:-60}"
  done
}
