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
GOVERNED_ENV_OVERRIDES = {
    "CREWAI_MAX_ITER": "1",
    "CREWAI_MAX_RETRY_LIMIT": "0",
    "MARKET_AGENT_PLANNER_TRANSIENT_RETRY_ATTEMPTS": "0",
    "MARKET_AGENT_PLANNER_TRANSIENT_RETRY_BASE_SECONDS": "0",
    "MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_ATTEMPTS": "0",
    "MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_BASE_SECONDS": "0",
}


@dataclass(frozen=True)
class CheckCommand:
    name: str
    command: tuple[str, ...]


def _check_commands() -> list[CheckCommand]:
    return [
        CheckCommand(
            name="semantic_keyword_guard",
            command=(sys.executable, "scripts/check_no_semantic_keyword_matching.py"),
        ),
        CheckCommand(
            name="prompt_registry",
            command=(sys.executable, "scripts/check_prompt_registry.py"),
        ),
        CheckCommand(
            name="runtime_fake_deps",
            command=(sys.executable, "scripts/check_reply_runtime_fake_deps.py"),
        ),
    ]


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
    child_environment = os.environ.copy()
    child_environment.update(GOVERNED_ENV_OVERRIDES)
    completed = subprocess.run(
        command.command,
        cwd=PROJECT_ROOT,
        env=child_environment,
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
    }
    if completed.returncode != 0 and not verbose:
        result["stdout_tail"] = _tail(completed.stdout)
        result["stderr_tail"] = _tail(completed.stderr)
        print(completed.stdout, file=sys.stdout)
        print(completed.stderr, file=sys.stderr)
    print(
        f"DONE {command.name}: {completed.returncode} in {duration_seconds}s",
        flush=True,
    )
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
        "--verbose",
        action="store_true",
        help="Stream child output instead of showing only failures and summary.",
    )
    args = parser.parse_args()

    _load_dotenv()
    results = [_run_command(command, args.verbose) for command in _check_commands()]
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
