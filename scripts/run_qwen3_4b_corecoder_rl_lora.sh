#!/bin/bash

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# keep stdout/stderr unbuffered in ray jobs
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

NUM_GPUS=${NUM_GPUS:-4}
ACTOR_GPUS=${ACTOR_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
PRM_GPUS=${PRM_GPUS:-1}

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${PRM_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT=${SLIME_ROOT:-/root/CoreCoder_RL/slime}
CORECODER_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="${CORECODER_ROOT}"

HF_CKPT=${HF_CKPT:-/root/autodl-tmp/corecoder_rl/models/Qwen3-4B}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-/root/autodl-tmp/corecoder_rl/ckpt/corecoder-qwen3-4b-lora}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-${HF_CKPT}}

export SGLANG_API_KEY="${SGLANG_API_KEY:-change-me}"
export SERVED_MODEL_NAME="qwen3-4b"
export HOST="0.0.0.0"
export PORT="30000"
export CORECODER_RECORD_ENABLED="${CORECODER_RECORD_ENABLED:-1}"  # 0=off, 1=on
export CORECODER_RECORD_FILE="${CORECODER_RECORD_FILE:-${REPO_ROOT}/results/corecoder_qwen3_4b_lora_record.jsonl}"
export CORECODER_METRICS_FILE="${CORECODER_METRICS_FILE:-${REPO_ROOT}/results/corecoder_rl_metrics.jsonl}"
export CORECODER_TRAIN_METRICS_FILE="${CORECODER_TRAIN_METRICS_FILE:-${REPO_ROOT}/results/corecoder_train_metrics.jsonl}"
export CORECODER_KEEP_LAST_CHECKPOINT_ONLY="${CORECODER_KEEP_LAST_CHECKPOINT_ONLY:-1}"
export TP="${TP:-1}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"
export PRM_M="${PRM_M:-3}"
export CORECODER_TEACHER_LOGPROB_ENABLED="${CORECODER_TEACHER_LOGPROB_ENABLED:-1}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-on_policy_distillation}"


CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 1
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path corecoder_rl.rollout.generate_rollout_corecoder

   --num-rollout 100000000
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}"
   --n-samples-per-prompt 1
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
   --rollout-temperature 0.6
   --reward-key score

   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
   --gradient-checkpointing
)

GRPO_ARGS=(
   --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

LORA_ARGS=(
   --use-lora
   --lora-rank 16
   --lora-alpha 32
   --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)

EVAL_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${TP}"
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static "${MEM_FRACTION_STATIC}"
   --sglang-context-length "${CONTEXT_LENGTH}"
   --sglang-reasoning-parser qwen3
)

PRM_ARGS=(
   --prm-enable
   --prm-num-gpus "${PRM_GPUS}"
   --prm-num-gpus-per-engine "${PRM_TP:-${TP}}"
   --prm-model-path "${PRM_MODEL_PATH}"
   --prm-m "${PRM_M}"
   --prm-temperature "${PRM_TEMPERATURE:-0.6}"
   --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
)

CUSTOM_ARGS=(
   --custom-generate-function-path corecoder_rl.server.generate
   --custom-rm-path corecoder_rl.server.reward_func
)

USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-corecoder_rl}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group qwen3-4b-corecoder-rl-lora
    --wandb-key ${WANDB_KEY_VALUE}
  )
else
  WANDB_ARGS=()
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SGLANG_API_KEY\": \"${SGLANG_API_KEY}\",
    \"HOST\": \"${HOST}\",
    \"PORT\": \"${PORT}\",
    \"SERVED_MODEL_NAME\": \"${SERVED_MODEL_NAME}\",
    \"CORECODER_RECORD_ENABLED\": \"${CORECODER_RECORD_ENABLED}\",
    \"CORECODER_RECORD_FILE\": \"${CORECODER_RECORD_FILE}\",
    \"CORECODER_METRICS_FILE\": \"${CORECODER_METRICS_FILE}\",
    \"CORECODER_TRAIN_METRICS_FILE\": \"${CORECODER_TRAIN_METRICS_FILE}\",
    \"CORECODER_KEEP_LAST_CHECKPOINT_ONLY\": \"${CORECODER_KEEP_LAST_CHECKPOINT_ONLY}\",
    \"PRM_M\": \"${PRM_M}\"
  }
}"

RAY_JOB_WAIT_ARGS=()
if [ "${RAY_JOB_NO_WAIT:-0}" = "1" ]; then
  RAY_JOB_WAIT_ARGS=(--no-wait)
fi

cd "${SLIME_ROOT}"
ray job submit --address="http://127.0.0.1:8265" \
   "${RAY_JOB_WAIT_ARGS[@]}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_ROOT}/train_async.py" \
   --train-backend fsdp \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]} \
   ${LORA_ARGS[@]}
