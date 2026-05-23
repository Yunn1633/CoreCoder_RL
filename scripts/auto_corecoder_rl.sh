#!/usr/bin/env bash
# Fully automated CoreCoder RL runner.
# Restarts training, watches metrics, and keeps feeding teacher-student data
# with an external DeepSeek student whenever the rollout server accepts samples.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CORECODER_ROOT=${CORECODER_ROOT:-$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)}
SLIME_ROOT=${SLIME_ROOT:-/root/CoreCoder_RL/slime}
CONDA_SH=${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-corecoder-rl}
RUN_SCRIPT=${RUN_SCRIPT:-${CORECODER_ROOT}/scripts/run_qwen3_4b_corecoder_rl_lora.sh}
LOG_DIR=${LOG_DIR:-/root/autodl-tmp/corecoder_rl/logs/corecoder_rl}
RESULTS_DIR=${RESULTS_DIR:-${CORECODER_ROOT}/results}
METRICS_FILE=${CORECODER_METRICS_FILE:-${RESULTS_DIR}/corecoder_rl_metrics.jsonl}
TRAIN_METRICS_FILE=${CORECODER_TRAIN_METRICS_FILE:-${RESULTS_DIR}/corecoder_train_metrics.jsonl}
FEEDER_LOG=${FEEDER_LOG:-${LOG_DIR}/auto_corecoder_rl_feeder.log}
SUPERVISOR_LOG=${SUPERVISOR_LOG:-${LOG_DIR}/auto_corecoder_rl_supervisor.log}
LAUNCH_LOG=${LAUNCH_LOG:-${LOG_DIR}/auto_corecoder_rl_launch.log}
DEEPSEEK_KEY_FILE=${DEEPSEEK_KEY_FILE:-/root/.corecoder_deepseek_api_key}

export CORECODER_ROOT SLIME_ROOT
export CORECODER_METRICS_FILE=${METRICS_FILE}
export CORECODER_TRAIN_METRICS_FILE=${TRAIN_METRICS_FILE}
export CORECODER_RECORD_ENABLED=${CORECODER_RECORD_ENABLED:-1}
export CORECODER_RECORD_FILE=${CORECODER_RECORD_FILE:-${RESULTS_DIR}/corecoder_qwen3_4b_lora_record.jsonl}
export RAY_JOB_NO_WAIT=${RAY_JOB_NO_WAIT:-1}
export USE_WANDB=${USE_WANDB:-0}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
export CORECODER_FEEDER_LIMIT=${CORECODER_FEEDER_LIMIT:-8}
export CORECODER_FEEDER_CYCLES=${CORECODER_FEEDER_CYCLES:-0}  # 0 means forever
export CORECODER_FEEDER_INTERVAL=${CORECODER_FEEDER_INTERVAL:-20}
export CORECODER_BOOT_WAIT_SECONDS=${CORECODER_BOOT_WAIT_SECONDS:-1200}
export CORECODER_RL_API_KEY=${CORECODER_RL_API_KEY:-${SGLANG_API_KEY:-change-me}}
export SGLANG_API_KEY=${SGLANG_API_KEY:-${CORECODER_RL_API_KEY}}
export CORECODER_RL_URL=${CORECODER_RL_URL:-http://127.0.0.1:30000/v1/chat/completions}
export CORECODER_STUDENT_MODE=${CORECODER_STUDENT_MODE:-deepseek}
export CORECODER_RL_METHOD=${CORECODER_RL_METHOD:-opsd}
export DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-https://api.deepseek.com}
export DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-flash}
export DEEPSEEK_STUDENT_MAX_TOKENS=${DEEPSEEK_STUDENT_MAX_TOKENS:-512}
export PYTHONPATH=${CORECODER_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
touch "${SUPERVISOR_LOG}" "${FEEDER_LOG}"

log() {
  local msg="[$(date '+%F %T')] $*"
  echo "${msg}" | tee -a "${SUPERVISOR_LOG}"
}

jsonl_count() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    wc -l < "${file}" | tr -d ' '
  else
    echo 0
  fi
}

latest_train_step() {
  local file="$1"
  if [[ ! -s "${file}" ]]; then
    echo none
    return 0
  fi
  python3 - "$file" <<'PY'
import json, sys
path = sys.argv[1]
step = None
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'step' in row:
            step = row['step']
print('none' if step is None else step)
PY
}

require_key() {
  if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "${DEEPSEEK_KEY_FILE}" ]]; then
    export DEEPSEEK_API_KEY="$(tr -d '\r\n' < "${DEEPSEEK_KEY_FILE}")"
  fi
  if [[ "${CORECODER_STUDENT_MODE}" == "deepseek" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    log "ERROR: DEEPSEEK_API_KEY is missing. Put it in env or ${DEEPSEEK_KEY_FILE}."
    exit 2
  fi
}

activate_env() {
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
}

restart_training() {
  log "restarting CoreCoder RL training; launch log: ${LAUNCH_LOG}"
  activate_env
  cd "${CORECODER_ROOT}"
  bash "${RUN_SCRIPT}" > "${LAUNCH_LOG}" 2>&1
  log "training launch command returned; Ray job should now be running asynchronously"
}

wait_for_port() {
  local deadline=$((SECONDS + CORECODER_BOOT_WAIT_SECONDS))
  log "waiting for CoreCoder RL API port 30000 to become reachable"
  while (( SECONDS < deadline )); do
    if python3 - <<'PY' >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(2)
s.connect(('127.0.0.1', 30000))
s.close()
PY
    then
      log "API port is reachable"
      return 0
    fi
    sleep 10
  done
  log "ERROR: API port did not become reachable before timeout"
  tail -n 80 "${LAUNCH_LOG}" | tee -a "${SUPERVISOR_LOG}" || true
  return 1
}

feed_once() {
  local before_samples before_train after_samples after_train before_step after_step status
  before_samples=$(jsonl_count "${METRICS_FILE}")
  before_train=$(jsonl_count "${TRAIN_METRICS_FILE}")
  before_step=$(latest_train_step "${TRAIN_METRICS_FILE}")
  log "feed cycle start: sample_rows=${before_samples}, train_rows=${before_train}, latest_step=${before_step}"

  set +e
  python3 "${CORECODER_ROOT}/corecoder_rl/automation/run_teacher_student_feeder.py" \
    --limit "${CORECODER_FEEDER_LIMIT}" \
    --student-mode "${CORECODER_STUDENT_MODE}" \
    --rl-method "${CORECODER_RL_METHOD}" \
    --output "${LOG_DIR}/teacher_student_${CORECODER_RL_METHOD}_auto_$(date '+%Y%m%d_%H%M%S').jsonl" \
    >> "${FEEDER_LOG}" 2>&1
  status=$?
  set -e

  after_samples=$(jsonl_count "${METRICS_FILE}")
  after_train=$(jsonl_count "${TRAIN_METRICS_FILE}")
  after_step=$(latest_train_step "${TRAIN_METRICS_FILE}")

  if [[ ${status} -ne 0 ]]; then
    log "feed cycle hit status=${status}; likely submission window is closed or model is still loading. Will retry."
    tail -n 20 "${FEEDER_LOG}" | sed 's/sk-[A-Za-z0-9_]*/sk-***REDACTED***/g' | tee -a "${SUPERVISOR_LOG}" || true
  else
    log "feed cycle ok: sample_rows ${before_samples}->${after_samples}, train_rows ${before_train}->${after_train}, step ${before_step}->${after_step}"
  fi
}

main_loop() {
  local cycle=0
  while true; do
    cycle=$((cycle + 1))
    if (( CORECODER_FEEDER_CYCLES > 0 && cycle > CORECODER_FEEDER_CYCLES )); then
      log "completed ${CORECODER_FEEDER_CYCLES} feeder cycles; leaving training running"
      break
    fi
    feed_once
    sleep "${CORECODER_FEEDER_INTERVAL}"
  done
}

main() {
  require_key
  log "auto CoreCoder RL supervisor started"
  log "metrics=${METRICS_FILE}; train_metrics=${TRAIN_METRICS_FILE}; feeder_limit=${CORECODER_FEEDER_LIMIT}; cycles=${CORECODER_FEEDER_CYCLES}; rl_method=${CORECODER_RL_METHOD}"
  restart_training
  wait_for_port
  main_loop
}

main "$@"
