from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_inventory_cli_rejects_reintroduced_legacy_recall_imports(
    tmp_path: Path,
) -> None:
    # Given: a clean repository copy whose retired recall imports are reintroduced.
    legacy_module = "market_support_crewai_agent." + "schemas"
    sandbox = tmp_path / "repo"
    _ = shutil.copytree(
        ROOT,
        sandbox,
        ignore=shutil.ignore_patterns(
            ".git", ".omo", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__"
        ),
    )
    flow = sandbox / "src/market_support_crewai_agent/runtime/recall/flow.py"
    _ = flow.write_text(
        flow.read_text(encoding="utf-8")
        + "\nfrom market_support_crewai_agent.runtime.policy.ontology import DomainContext\n"
        + "_legacy_context: DomainContext\n",
        encoding="utf-8",
    )
    integration = sandbox / "tests/integration/runtime/test_question_recall_runtime.py"
    _ = integration.write_text(
        integration.read_text(encoding="utf-8")
        + f"\nfrom {legacy_module} import ReplyRequest\n",
        encoding="utf-8",
    )

    # When: the inventory command scans the reintroduced source flow.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(sandbox / "scripts/check_request_consumer_migration.py"),
            "--inventory",
        ],
        cwd=sandbox,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: the undeclared imports produce inventory drift and no success marker.
    assert completed.returncode == 1
    assert "request_consumer_inventory_drift" in completed.stderr
    assert "OK:" not in completed.stdout
