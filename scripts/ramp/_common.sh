#!/usr/bin/env bash
# Shared helpers for RAMP reproduction/evaluation scripts.

repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$here/../.." && pwd
}

ramp_checkpoint_for_seed() {
  local seed="${1:?seed required}"
  printf '%s/external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_%s/ep_80_0.pth' \
    "$(repo_root)" "$seed"
}

require_ramp_checkout() {
  local root
  root="$(repo_root)"
  if [[ ! -d "$root/external/RAMP" ]]; then
    echo "ERROR: external/RAMP is missing. Clone https://github.com/uiuc-focal-lab/RAMP first." >&2
    exit 1
  fi
}

require_checkpoint() {
  local ckpt="${1:?checkpoint required}"
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: RAMP checkpoint not found: $ckpt" >&2
    echo "Train it first, e.g. bash scripts/ramp/train_ramp_seed0.sh" >&2
    exit 1
  fi
}

python_bin_from_repo() {
  local root
  root="$(repo_root)"
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
  elif [[ -x "$root/.venv/bin/python" ]]; then
    printf '%s\n' "$root/.venv/bin/python"
  else
    printf '%s\n' "python"
  fi
}

check_ramp_training_deps() {
  local py="${1:?python executable required}"
  if ! "$py" -c "import robustbench" >/dev/null 2>&1; then
    echo "ERROR: RAMP.py imports robustbench, but it is not available for: $py" >&2
    echo "Install RAMP's dependencies before training, e.g. from external/RAMP/requirements.txt." >&2
    echo "No training was started." >&2
    exit 1
  fi
}
