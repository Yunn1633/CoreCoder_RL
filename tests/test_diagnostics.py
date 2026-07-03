from corecoder_rl.diagnostics import analyze_metrics, analyze_records, render_report


def test_analyze_metrics_counts_rollout_quality_signals():
    rows = [
        {
            "session_id": "s1",
            "turn": 1,
            "response_tokens": 10,
            "prompt_tokens": 100,
            "reward": 1.0,
            "loss_mask": 1,
            "has_next_state": True,
            "teacher_extra_prompt_present": True,
            "prm_votes": [1],
            "step_action_type": "tool_call",
            "step_scoring_source": "prm",
        },
        {
            "session_id": "s1",
            "turn": 2,
            "response_tokens": 20,
            "prompt_tokens": 120,
            "reward": 0.0,
            "loss_mask": 0,
            "has_next_state": False,
            "step_action_type": "final_answer",
        },
    ]

    stats = analyze_metrics(rows)

    assert stats["rows"] == 2
    assert stats["sessions"] == 1
    assert stats["effective_samples"] == 1
    assert stats["no_next_state"] == 1
    assert stats["zero_reward"] == 1
    assert stats["teacher_extra"] == 1
    assert stats["prm_rows"] == 1
    assert stats["action_types"]["tool_call"] == 1


def test_render_report_handles_record_only_input():
    records = [
        {"session_id": "s1", "response_text": "hello", "next_state": None},
        {"session_id": "s2", "response_text": "tool", "tool_calls": [{"name": "python"}], "next_state": {"role": "tool"}},
    ]

    stats = analyze_records(records)
    report = render_report([], records, num_gpus=1)

    assert stats["rows"] == 2
    assert stats["next_state_missing"] == 1
    assert stats["tool_rows"] == 1
    assert "Single GPU: do not run full train_async" in report
    assert "Records" in report
