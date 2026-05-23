# Architecture Notes

CoreCoder_RL has three moving parts:

1. `corecoder_rl.server.CoreCoderAPIServer` proxies OpenAI-style chat requests to SGLang and turns main assistant messages into slime `Sample` objects.
2. `corecoder_rl.automation.run_teacher_student_feeder` drives multi-turn teacher-student sessions. DeepSeek Flash can role-play the student and, in OPSD mode, produce the extra student-side prompt.
3. The slime FSDP overlay carries `teacher_log_probs` through rollout packing and optimizes an on-policy distillation loss.

The training loss is continuous because it is computed from token-level logprob differences, not from the discrete PRM reward. PRM is still recorded as process feedback and can be used for filtering/masking.
