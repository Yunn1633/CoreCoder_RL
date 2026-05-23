# CoreCoder_RL

[English](README.md) | [中文](README_CN.md)

CoreCoder_RL is a lightweight research project for agentic RL training built on [slime](https://github.com/THUDM/slime) and [CoreCoder](https://github.com/he-yufeng/CoreCoder).

## Project Contents

- `corecoder_rl/`: CoreCoder RL proxy service, rollout, and automated training code.
- `scripts/`: training scripts.
- `patches/corecoder_slime_overlay/`: FSDP training overlay.

## OPSD Training Flow

- The policy model samples on-policy responses through the CoreCoder RL proxy.
- The teacher model observes the same generated student trajectory and can use the question, optional reference answer, tool information, and grading rubric to generate an extra prompt.
- The teacher computes `teacher_log_probs` for the tokens in the student trajectory.
- The training objective uses the student/teacher logprob difference on the student trajectory, encouraging the policy model to move closer to the teacher's judgment on the same trajectory.

## Tool-use Module

CoreCoder_RL supports tool-use training.

It implements a simple experiment based on a Python tool: training data is randomly sampled from GSM8K, and the student model can call Python when it is helpful.

## Example Result

![loss](docs/current_opsd_tooluse_loss_ema.png)

## License

The CoreCoder_RL glue code in this repository is released under the MIT License. Upstream projects and copied/overlaid files remain governed by their original licenses. Please review the licenses of CoreCoder, slime, SGLang, Ray, Transformers, PyTorch, and the relevant models before redistribution or commercial use.

## Acknowledgements

This project is built upon the excellent work of [CoreCoder](https://github.com/he-yufeng/CoreCoder), [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL), and [slime](https://github.com/THUDM/slime). We thank the authors of these projects.
