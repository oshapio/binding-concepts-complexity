#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_analyses.sh --lrs 1e-3,1e-4 --embeddings /path/a.pt,/path/b.pt [--profiles regular_mlp,mult_linear,sum_mult_linear] [--regular-hidden-specs 128,256,512] [--python python]

Runs approximate_complexity_scenes.py across:
- training fractions: 0.1..0.9
- profile-specific probe/MLP settings

Args:
  --lrs         Comma-separated learning rates (required)
  --embeddings  Comma-separated embeddings paths (required)
  --profiles    Comma-separated profile names (default: regular_mlp)
                - regular_mlp:    non-multiplicative concat MLP + hidden-layer sweep from --regular-hidden-specs
                - mult_linear:    multiplicative probe only, linear head
                - sum_mult_linear:multiplicative + sum probe, linear head
  --regular-hidden-specs
                Comma-separated hidden widths for regular_mlp (default: 128,256,512,1024,2048,4096)
  --python      Python executable (default: python)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
LRS_CSV=""
EMB_CSV=""
PROFILES_CSV="regular_mlp"
REGULAR_HIDDEN_SPECS_CSV="128,256,512,1024,2048,4096"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lrs)
      LRS_CSV="${2:-}"
      shift 2
      ;;
    --embeddings)
      EMB_CSV="${2:-}"
      shift 2
      ;;
    --profiles)
      PROFILES_CSV="${2:-}"
      shift 2
      ;;
    --regular-hidden-specs)
      REGULAR_HIDDEN_SPECS_CSV="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

if [[ -z "$LRS_CSV" || -z "$EMB_CSV" ]]; then
  echo "Missing required --lrs and/or --embeddings." >&2
  usage
  exit 1
fi

IFS=',' read -r -a LR_LIST <<< "$LRS_CSV"
IFS=',' read -r -a EMB_LIST <<< "$EMB_CSV"
IFS=',' read -r -a PROFILE_LIST <<< "$PROFILES_CSV"
IFS=',' read -r -a REGULAR_HIDDEN_SPECS <<< "$REGULAR_HIDDEN_SPECS_CSV"

if [[ ${#REGULAR_HIDDEN_SPECS[@]} -eq 0 ]]; then
  echo "--regular-hidden-specs must provide at least one width." >&2
  exit 1
fi

TRAIN_FRACTIONS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

APPROX_SCRIPT="${SCRIPT_DIR}/approximate_complexity_scenes.py"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

profile_config() {
  local profile="$1"
  case "$profile" in
    regular_mlp)
      PROFILE_HIDDEN_SPECS=("${REGULAR_HIDDEN_SPECS[@]}")
      PROFILE_ARGS=(--mult_probes false --sum_and_mult false --use_W_mult false --mult_within_obj false)
      ;;
    mult_linear)
      PROFILE_HIDDEN_SPECS=("none")
      PROFILE_ARGS=(--mult_probes true --sum_and_mult false --use_W_mult false --mult_within_obj false)
      ;;
    sum_mult_linear)
      PROFILE_HIDDEN_SPECS=("none")
      PROFILE_ARGS=(--mult_probes true --sum_and_mult true --use_W_mult false --mult_within_obj false)
      ;;
    *)
      echo "Unknown profile: $profile" >&2
      echo "Supported profiles: regular_mlp, mult_linear, sum_mult_linear" >&2
      exit 1
      ;;
  esac
}

for emb in "${EMB_LIST[@]}"; do
  emb="$(trim "$emb")"
  for lr in "${LR_LIST[@]}"; do
    lr="$(trim "$lr")"
    for frac in "${TRAIN_FRACTIONS[@]}"; do
      for profile in "${PROFILE_LIST[@]}"; do
        profile="$(trim "$profile")"
        profile_config "$profile"
        for hidden_spec in "${PROFILE_HIDDEN_SPECS[@]}"; do
          hidden_spec="$(trim "$hidden_spec")"

          hidden_args=()
          if [[ "$hidden_spec" == "none" ]]; then
            # Empty hidden layer list: pass flag with no values.
            hidden_args=(--hidden-layers)
          else
            # Accept either comma-separated ("1024,1024") or space-separated ("1024 1024") specs.
            hidden_spec_norm="${hidden_spec// /,}"
            IFS=',' read -r -a hidden_layers <<< "$hidden_spec_norm"
            hidden_args=(--hidden-layers "${hidden_layers[@]}")
          fi

          echo "Running emb=${emb} lr=${lr} train_fraction=${frac} profile=${profile} hidden_layers=${hidden_spec}"
          "$PYTHON_BIN" "$APPROX_SCRIPT" \
            --embeddings-path "$emb" \
            --lr_use "$lr" \
            --train-fraction "$frac" \
            "${PROFILE_ARGS[@]}" \
            "${hidden_args[@]}"
        done
      done
    done
  done
done
