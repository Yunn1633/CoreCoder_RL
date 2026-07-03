# Rollout Doctor

`corecoder-rl doctor` is an offline diagnostic command for CoreCoder_RL rollout logs.
It does not start Ray, SGLang, PRM, or training.

Use it on single-GPU or CPU-only machines to inspect whether rollout throughput is
limited by sample production, missing next states, zero rewards, PRM, teacher extra
prompt generation, or long multi-turn sessions.

## Usage

```bash
python -m corecoder_rl.cli doctor \
  --metrics results/corecoder_rl_metrics.jsonl \
  --records results/corecoder_qwen3_4b_lora_record.jsonl \
  --num-gpus 1
```

If the package entry point is installed, this is equivalent:

```bash
corecoder-rl doctor \
  --metrics results/corecoder_rl_metrics.jsonl \
  --records results/corecoder_qwen3_4b_lora_record.jsonl \
  --num-gpus 1
```

## What It Reports

- session count and turns per session
- average and p95 prompt/response token lengths
- effective sample ratio from `loss_mask`
- missing `next_state` ratio
- zero-reward ratio
- PRM and teacher extra prompt usage
- action type and scoring source breakdown
- single-GPU recommendations

## Interpretation

High missing `next_state` or zero-reward ratios usually point to feeder/session
finalization or scoring issues. High turn p95 means large `rollout-batch-size`
will amplify long-tail waiting. If teacher extra prompt or PRM appears on most
rows, disable those first when measuring baseline rollout throughput.
