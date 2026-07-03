from __future__ import annotations

import argparse
import json
from pathlib import Path

from corecoder_rl.diagnostics import add_doctor_args, run_doctor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="corecoder-rl")
    sub = parser.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan")
    export = sub.add_parser("export-records")
    export.add_argument("record_file")
    doctor = sub.add_parser("doctor")
    add_doctor_args(doctor)
    args = parser.parse_args(argv)
    if args.cmd == "plan":
        print("CoreCoder RL uses CoreCoder as the runtime/API client boundary and SLIME train_async.py as the trainer.")
        print("Run: conda activate corecoder-rl && cd /root/CoreCoder && bash scripts/run_qwen3_4b_corecoder_rl_lora.sh")
        print("Send OpenAI-compatible traffic to http://127.0.0.1:30000/v1/chat/completions with session_id/turn_type/session_done.")
        return 0
    if args.cmd == "doctor":
        return run_doctor(args)
    for line in Path(args.record_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            print(json.dumps(json.loads(line), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
