from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CheckCommand:
    name: str
    command: tuple[str, ...]
    requires_real_llm: bool = False
    live_adapter: bool = False


def _check_commands(
    *, include_real_llm: bool, include_live_adapter: bool
) -> list[CheckCommand]:
    commands = [
        CheckCommand(
            name="runtime_fake_deps",
            command=(sys.executable, "scripts/check_reply_runtime_fake_deps.py"),
        ),
    ]
    if include_real_llm:
        commands.extend(
            [
                CheckCommand(
                    name="real_llm_knowledge_eval",
                    command=(sys.executable, "scripts/eval_reply_real_llm_knowledge.py"),
                    requires_real_llm=True,
                ),
                CheckCommand(
                    name="real_llm_action_eval",
                    command=(sys.executable, "scripts/eval_reply_real_llm_actions.py"),
                    requires_real_llm=True,
                ),
                CheckCommand(
                    name="handoff_eval",
                    command=(sys.executable, "scripts/eval_reply_handoff.py"),
                    requires_real_llm=True,
                ),
                CheckCommand(
                    name="compliance_eval",
                    command=(sys.executable, "scripts/eval_reply_compliance.py"),
                    requires_real_llm=True,
                ),
                CheckCommand(
                    name="action_feedback_ledger",
                    command=(sys.executable, "scripts/check_reply_action_feedback.py"),
                    requires_real_llm=True,
                ),
            ]
        )
    if include_live_adapter:
        commands.append(
            CheckCommand(
                name="live_adapter_eval",
                command=(
                    sys.executable,
                    "scripts/eval_reply_live_adapter.py",
                    "--message",
                    "请发一下周报",
                ),
                requires_real_llm=True,
                live_adapter=True,
            )
        )
    return commands


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _run_command(command: CheckCommand, verbose: bool) -> dict:
    print(f"RUN {command.name}: {' '.join(command.command)}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command.command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=not verbose,
        check=False,
    )
    duration_seconds = round(time.perf_counter() - started, 3)
    result = {
        "name": command.name,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": duration_seconds,
        "requires_real_llm": command.requires_real_llm,
        "live_adapter": command.live_adapter,
    }
    if completed.returncode != 0 and not verbose:
        result["stdout_tail"] = _tail(completed.stdout)
        result["stderr_tail"] = _tail(completed.stderr)
        print(completed.stdout, file=sys.stdout)
        print(completed.stderr, file=sys.stderr)
    print(f"DONE {command.name}: {completed.returncode} in {duration_seconds}s", flush=True)
    return result


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the support-reply acceptance check suite."
    )
    parser.add_argument(
        "--include-real-llm",
        action="store_true",
        help="Also run eval scripts that call the configured LLM provider.",
    )
    parser.add_argument(
        "--include-live-adapter",
        action="store_true",
        help="Also run the live adapter eval. Requires xiaoyan adapter or fixture to be running.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream child output instead of showing only failures and summary.",
    )
    args = parser.parse_args()

    _load_dotenv()
    commands = _check_commands(
        include_real_llm=args.include_real_llm,
        include_live_adapter=args.include_live_adapter,
    )

    results = [_run_command(command, args.verbose) for command in commands]
    failures = [result for result in results if not result["passed"]]
    summary = {
        "passed": not failures,
        "total": len(results),
        "failed": len(failures),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
