from __future__ import annotations

import argparse
import sys
from pathlib import Path

from market_support_crewai_agent.runtime.state.latency_report import (
    format_latency_report,
    latency_report_from_lines,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize /reply runtime_trace span latency from agent_runtime_trace logs."
    )
    parser.add_argument("path", nargs="?", help="Log file path. Reads stdin when omitted.")
    parser.add_argument("--limit", type=int, default=20, help="Recent trace count to include.")
    parser.add_argument("--top", type=int, default=15, help="Span rows to print.")
    args = parser.parse_args()

    if args.path:
        lines = _read_log_text(Path(args.path)).splitlines()
    else:
        lines = sys.stdin.read().splitlines()
    report = latency_report_from_lines(lines, limit=args.limit)
    print(format_latency_report(report, top=args.top))
    return 0


def _read_log_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
