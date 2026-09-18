from __future__ import annotations

import hashlib

# noqa: SIZE_OK
# pyright: reportAny=false, reportUnusedCallResult=false
import json
import subprocess
from pathlib import Path

import pytest

from scripts.check_request_consumer_migration import (
    collect_git_baseline,
    scan_consumers,
)

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/check_request_consumer_migration.py"

PLANNED_STRUCTURAL_SHA256 = (
    "82554239fe2d05b2cc191ff15b20a5bd4cceb3dd1a704ac8ba78ae78802292dc"
)
TODO_1_14_PREFIX_SHA256 = (
    "d5bb481164645b1feeb6a9fb13d10ae3e856787e124ddc868729322dfa49928c"
)
DEFERRED_SYMBOL_ROWS: tuple[dict[str, str | int], ...] = (
    {
        "symbol": "AdapterDirectAttestationV1",
        "path": "src/market_support_crewai_agent/schemas.py",
        "target_todo": 15,
        "kind": "class",
        "expected_count": 1,
        "status": "planned_boundary",
    },
    {
        "symbol": "DirectReadinessHealthV1",
        "path": "src/market_support_crewai_agent/schemas.py",
        "target_todo": 15,
        "kind": "class",
        "expected_count": 1,
        "status": "planned_boundary",
    },
    {
        "symbol": "AgentBuildFileRecordV1",
        "path": "src/market_support_crewai_agent/runtime/build_metadata.py",
        "target_todo": 15,
        "kind": "class",
        "expected_count": 1,
        "status": "planned_boundary",
    },
    {
        "symbol": "AgentBuildMetadataV1",
        "path": "src/market_support_crewai_agent/runtime/build_metadata.py",
        "target_todo": 15,
        "kind": "class",
        "expected_count": 1,
        "status": "planned_boundary",
    },
    {
        "symbol": "LateImportGuardV1",
        "path": "src/market_support_crewai_agent/runtime/build_metadata.py",
        "target_todo": 15,
        "kind": "class",
        "expected_count": 1,
        "status": "planned_boundary",
    },
    {
        "symbol": "DirectAttestationNonceStoreV1",
        "path": "src/market_support_crewai_agent/runtime/integrations/adapter/preflight.py",
        "target_todo": 15,
        "kind": "class",
        "expected_count": 1,
        "status": "planned_boundary",
    },
    {
        "symbol": "FakeScenarioV2",
        "path": "scripts/scene_http_control_models.py",
        "target_todo": 16,
        "kind": "class",
        "expected_count": 1,
        "status": "planned_boundary",
    },
    {
        "symbol": "FakeScenarioArmedV2",
        "path": "scripts/scene_http_control_models.py",
        "target_todo": 16,
        "kind": "class",
        "expected_count": 1,
        "status": "planned_boundary",
    },
    {
        "symbol": "FakeControlStateV2",
        "path": "scripts/scene_http_control_models.py",
        "target_todo": 16,
        "kind": "class",
        "expected_count": 1,
        "status": "planned_boundary",
    },
    {
        "symbol": "FakeBarrierReleaseV1",
        "path": "scripts/scene_http_control_models.py",
        "target_todo": 16,
        "kind": "class",
        "expected_count": 1,
        "status": "planned_boundary",
    },
)


def _write_minimal_planned_sources(root: Path) -> None:
    source = ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    by_path: dict[str, list[dict[str, str | int]]] = {}
    for row in payload["rows"]:
        by_path.setdefault(str(row["path"]), []).append(row)
    for relative, rows in by_path.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = ["from __future__ import annotations", ""]
        for row in rows:
            symbol = str(row["symbol"])
            kind = row["kind"]
            if kind == "class":
                lines.append(f"class {symbol}:")
                lines.append("    pass")
            elif kind in {"function", "method"}:
                lines.append(f"def {symbol}() -> None:")
                lines.append("    return None")
            else:
                lines.append(f"{symbol} = object")
            lines.append("")
        target.write_text("\n".join(lines), encoding="utf-8")


def _planned_structural_digest(rows: list[dict[str, str | int]]) -> str:
    structural_rows = [
        {
            key: row[key]
            for key in ("symbol", "path", "target_todo", "kind", "expected_count")
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            structural_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_phase_cli_rejects_raw_legacy_source_consumer_without_relabeling(
    tmp_path: Path,
) -> None:
    # Given: a disposable source tree whose scanner and fixture both contain one raw legacy row.
    _write_minimal_planned_sources(tmp_path)
    consumer = tmp_path / "src/legacy_consumer.py"
    consumer.write_text(
        "\n".join(
            (
                "from market_support_crewai_agent.runtime.planning.models import ExecutionPlan",
                "def consume(plan: ExecutionPlan) -> ExecutionPlan:",
                "    return plan",
            )
        ),
        encoding="utf-8",
    )
    fixture = tmp_path / "request-consumer.json"
    inventory = scan_consumers(tmp_path)
    assert {row.status for row in inventory.rows} == {"legacy"}
    fixture.write_text(inventory.model_dump_json(), encoding="utf-8")

    # When: phase validation checks the raw migration inventory.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--phase",
            "6",
            "--root",
            str(tmp_path),
            "--fixture",
            str(fixture),
            "--planned-fixture",
            str(ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: target_todo cannot reclassify raw legacy rows into a passing phase.
    assert completed.returncode == 1
    assert "phase_has_pending_consumer_rows" in completed.stderr
    assert "OK:" not in completed.stdout


def test_require_complete_cli_rejects_raw_planned_boundary_fixture_row(
    tmp_path: Path,
) -> None:
    # Given: a schema-valid fixture with a raw planned-boundary consumer row.
    _write_minimal_planned_sources(tmp_path)
    fixture = tmp_path / "request-consumer.json"
    fixture.write_text(
        json.dumps(
            {
                "contract_version": "request-consumer-migration.v1",
                "rows": [
                    {
                        "path": "src/generated_boundary.py",
                        "symbol": "GeneratedBoundary",
                        "match_kind": "manual",
                        "expected_count": 1,
                        "target_todo": 6,
                        "target_type": "ReplyRequestV2",
                        "status": "planned_boundary",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # When: completion validation inspects the raw fixture status.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--require-complete",
            "--root",
            str(tmp_path),
            "--fixture",
            str(fixture),
            "--planned-fixture",
            str(ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: planned-boundary rows are never hidden behind success output.
    assert completed.returncode == 1
    assert "ERROR:" in completed.stderr
    assert "OK:" not in completed.stdout


@pytest.mark.parametrize("status", ("planned_boundary", "legacy"))
def test_require_complete_cli_rejects_raw_planned_symbol_status(
    tmp_path: Path,
    status: str,
) -> None:
    # Given: the planned-symbol fixture keeps valid structure but one raw pending status.
    source = ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    pending_row = payload["rows"][0]
    pending_row["status"] = status
    planned = tmp_path / f"planned-symbol-{status}.json"
    planned.write_text(json.dumps(payload), encoding="utf-8")

    # When: completion validation inspects both consumer rows and planned rows.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--require-complete",
            "--planned-fixture",
            str(planned),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: a pending planned symbol fails closed despite an unchanged structural digest.
    assert completed.returncode == 1
    assert (
        f"migration_inventory_incomplete:planned_symbol:{pending_row['symbol']}:{status}"
        in completed.stderr
    )
    assert "OK:" not in completed.stdout


def test_require_complete_cli_accepts_terminal_all_migrated_fixtures() -> None:
    # Given: the resealed terminal consumer and planned-symbol fixtures.
    consumer = ROOT / "tests/fixtures/request_consumer_migration.v1.json"
    planned = ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"

    # When: completion validation runs against the authoritative fixtures.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--require-complete",
            "--fixture",
            str(consumer),
            "--planned-fixture",
            str(planned),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: the real all-migrated 9/80 terminal state remains accepted.
    assert completed.returncode == 0
    assert "OK: 7 consumer rows" in completed.stdout
    assert "80 planned symbols" in completed.stdout


def test_require_complete_cli_rejects_raw_legacy_at_former_retained_key(
    tmp_path: Path,
) -> None:
    # Given: a raw legacy consumer at the former hard-coded retained-key path.
    _write_minimal_planned_sources(tmp_path)
    probe = tmp_path / "src/market_support_crewai_agent/runtime/recall/models.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        "from __future__ import annotations\n"
        + "\n".join(
            (
                "from market_support_crewai_agent.runtime.planning.models import ExecutionPlan",
                "def execution_plan_for_question_recall() -> ExecutionPlan:",
                "    return ExecutionPlan()",
            )
        ),
        encoding="utf-8",
    )
    fixture = tmp_path / "former-retained-key.json"
    inventory = scan_consumers(tmp_path)
    assert {row.status for row in inventory.rows} == {"legacy"}
    fixture.write_text(inventory.model_dump_json(), encoding="utf-8")

    # When: completion validation checks the generated raw fixture.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--require-complete",
            "--root",
            str(tmp_path),
            "--fixture",
            str(fixture),
            "--planned-fixture",
            str(ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: the former retained-key relabel path cannot print success.
    assert completed.returncode == 1
    assert "migration_inventory_incomplete" in completed.stderr
    assert "OK:" not in completed.stdout


def test_inventory_cli_rejects_malformed_in_scope_python_source(
    tmp_path: Path,
) -> None:
    # Given: an in-scope Python source file that the scanner cannot parse.
    broken = tmp_path / "src/malformed_gate_probe.py"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("def broken(:\n", encoding="utf-8")

    # When: inventory validation scans that source tree.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--inventory",
            "--root",
            str(tmp_path),
            "--fixture",
            str(ROOT / "tests/fixtures/request_consumer_migration.v1.json"),
            "--planned-fixture",
            str(ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: parse errors fail closed before any OK marker.
    assert completed.returncode == 1
    assert "scan_source_parse_error:src/malformed_gate_probe.py:1" in completed.stderr
    assert "OK:" not in completed.stdout


def test_amendment_fixture_reseals_todo_1_14_prefix_and_replaces_deferred_rows() -> (
    None
):
    # Given: the authoritative planned-symbol ownership fixture.
    source = ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload["rows"]

    # When: the amendment boundary is inspected structurally.
    prefix = [row for row in rows if row["target_todo"] <= 14]
    target_rows = [row for row in rows if row["target_todo"] == 15]
    structural_digest = _planned_structural_digest(rows)

    # Then: the 77-row prefix and exact three new boundary rows are sealed.
    assert len(rows) == 80
    assert len(prefix) == 77
    assert _planned_structural_digest(prefix) == TODO_1_14_PREFIX_SHA256
    assert structural_digest == PLANNED_STRUCTURAL_SHA256
    assert target_rows == [
        {
            "symbol": "DeploymentIdentityError",
            "path": "src/market_support_crewai_agent/runtime/identity/deployment.py",
            "target_todo": 15,
            "kind": "class",
            "expected_count": 1,
            "status": "migrated",
        },
        {
            "symbol": "validate_deployment_identity",
            "path": "src/market_support_crewai_agent/runtime/identity/deployment.py",
            "target_todo": 15,
            "kind": "function",
            "expected_count": 1,
            "status": "migrated",
        },
        {
            "symbol": "assert_scene_compatible",
            "path": "src/market_support_crewai_agent/runtime/integrations/adapter/client.py",
            "target_todo": 15,
            "kind": "method",
            "expected_count": 1,
            "status": "migrated",
        },
    ]
    assert not any(row["target_todo"] == 16 for row in rows)
    assert not {(row["symbol"], row["path"]) for row in rows}.intersection(
        {(row["symbol"], row["path"]) for row in DEFERRED_SYMBOL_ROWS}
    )


@pytest.mark.parametrize("deferred_row", DEFERRED_SYMBOL_ROWS)
def test_inventory_rejects_reinserted_deferred_boundary_row(
    tmp_path: Path,
    deferred_row: dict[str, str | int],
) -> None:
    # Given: a copy of the sealed fixture with one deferred row reinserted.
    source = ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["rows"].append(deferred_row)
    mutated = tmp_path / f"reinsert-{deferred_row['symbol']}.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    # When: the inventory command validates the mutated copy.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--inventory",
            "--planned-fixture",
            str(mutated),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: the sealed structural boundary fails closed without an OK marker.
    assert completed.returncode == 1
    assert (
        "planned_symbol_ownership_drift" in completed.stderr
        or "duplicate_planned_symbol_row" in completed.stderr
    )
    assert "OK:" not in completed.stdout


def test_baseline_collector_handles_committed_and_unborn_repositories(
    tmp_path: Path,
) -> None:
    # Given: one committed repository and one unborn repository.
    committed = tmp_path / "committed"
    unborn = tmp_path / "unborn"
    for repository in (committed, unborn):
        repository.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (committed / "tracked.txt").write_text("baseline", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=committed, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Todo One",
            "-c",
            "user.email=todo@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=committed,
        check=True,
    )
    (unborn / "untracked.txt").write_text("unborn", encoding="utf-8")
    # When: both repositories are collected through the same read-only boundary.
    committed_result = collect_git_baseline(committed)
    unborn_result = collect_git_baseline(unborn)

    # Then: committed HEAD is recorded and unborn collection never requires one.
    assert committed_result.head is not None
    assert unborn_result.head is None
    assert b"untracked.txt" in unborn_result.status
    assert unborn_result.rows[0].path == "untracked.txt"
    assert unborn_result.rows[0].record_kind == "?"


def test_inventory_cli_rejects_malformed_fixture(tmp_path: Path) -> None:
    # Given: fixture paths containing invalid inventory shapes.
    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        '{"contract_version":"request-consumer-migration.v1","rows":[{"path":"x"}]}',
        encoding="utf-8",
    )

    # When: inventory validation is invoked against the malformed boundary.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--inventory",
            "--fixture",
            str(malformed),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: the command is bounded and fails without a success marker.
    assert completed.returncode == 1
    assert "ERROR:" in completed.stderr
    assert "OK:" not in completed.stdout


def test_baseline_collector_preserves_every_dirty_status_class(tmp_path: Path) -> None:
    # Given: a disposable repository with staged, unstaged, renamed, deleted, and untracked paths.
    repository = tmp_path / "dirty"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    for name in ("staged.txt", "unstaged.txt", "renamed.txt", "deleted.txt"):
        (repository / name).write_text("baseline", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Todo One",
            "-c",
            "user.email=todo@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=repository,
        check=True,
    )
    (repository / "staged.txt").write_text("staged", encoding="utf-8")
    subprocess.run(["git", "add", "staged.txt"], cwd=repository, check=True)
    (repository / "unstaged.txt").write_text("unstaged", encoding="utf-8")
    subprocess.run(
        ["git", "mv", "renamed.txt", "renamed-new.txt"], cwd=repository, check=True
    )
    (repository / "deleted.txt").unlink()
    (repository / "untracked.txt").write_text("untracked", encoding="utf-8")
    # When: the dirty tree is captured without applying any Git mutation.
    before = collect_git_baseline(repository)
    after = collect_git_baseline(repository)

    # Then: every class remains represented and the capture is byte-stable.
    assert before == after
    assert b"staged.txt" in before.status
    assert b"unstaged.txt" in before.status
    assert b"renamed-new.txt" in before.status
    assert b"deleted.txt" in before.status
    assert b"untracked.txt" in before.status
    by_path = {row.path: row for row in before.rows}
    assert by_path["staged.txt"].index_status == "M"
    assert by_path["unstaged.txt"].worktree_status == "M"
    assert by_path["renamed-new.txt"].record_kind == "2"
    assert by_path["renamed-new.txt"].original_path == "renamed.txt"
    assert by_path["deleted.txt"].worktree_status == "D"
    assert by_path["untracked.txt"].record_kind == "?"


@pytest.mark.parametrize(
    ("field", "value"),
    (("expected_count", 0), ("target_todo", 17), ("target_type", 7)),
)
def test_inventory_cli_rejects_invalid_count_owner_or_type(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    # Given: one otherwise valid row with a malformed count, owner, or type.
    fixture = tmp_path / "invalid.json"
    row = {
        "path": "src/example.py",
        "symbol": "ReplyRequest",
        "match_kind": "annotation",
        "expected_count": 1,
        "target_todo": 2,
        "target_type": "KernelReplyRequestV1",
        "status": "legacy",
    }
    row[field] = value
    fixture.write_text(
        json.dumps(
            {"contract_version": "request-consumer-migration.v1", "rows": [row]}
        ),
        encoding="utf-8",
    )

    # When: the typed inventory boundary parses the malformed fixture.
    completed = subprocess.run(
        ["uv", "run", "python", str(SCRIPT), "--inventory", "--fixture", str(fixture)],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: validation fails before any misleading success output.
    assert completed.returncode == 1
    assert "ERROR:" in completed.stderr
    assert "OK:" not in completed.stdout


def test_inventory_cli_detects_stale_valid_fixture_copy(tmp_path: Path) -> None:
    # Given: a disposable, schema-valid copy with one semantic status mutation.
    source = ROOT / "tests/fixtures/request_consumer_migration.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["rows"][0]["status"] = "legacy"
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(payload), encoding="utf-8")

    # When: the scanner compares live AST flow to the stale copy.
    completed = subprocess.run(
        ["uv", "run", "python", str(SCRIPT), "--inventory", "--fixture", str(stale)],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: drift fails and the authoritative fixture remains untouched.
    assert completed.returncode == 1
    assert "request_consumer_inventory_drift" in completed.stderr
    assert (
        json.loads(source.read_text(encoding="utf-8"))["rows"][0]["status"]
        == "migrated"
    )


@pytest.mark.parametrize("mutation", ("reclassify", "remove"))
def test_phase8_rejects_recall_consumer_row_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    # Given: a Todo-8 fixture with one live recall consumer row changed.
    source = ROOT / "tests/fixtures/request_consumer_migration.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    target = payload["rows"][0]
    if mutation == "reclassify":
        target["status"] = "legacy"
    else:
        payload["rows"].remove(target)
    mutated = tmp_path / "recall-boundary-mutation.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    # When: phase-8 validation compares the mutated fixture against live source flow.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--phase",
            "8",
            "--fixture",
            str(mutated),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: the checker reports drift instead of normalizing away raw status/counts.
    assert completed.returncode == 1
    assert "request_consumer_inventory_drift" in completed.stderr


def test_inventory_cli_allows_two_consumers_for_one_target_definition(
    tmp_path: Path,
) -> None:
    # Given: two retained consumer occurrences targeting one canonical replacement class.
    _write_minimal_planned_sources(tmp_path)
    replacement = tmp_path / "src/replacement.py"
    replacement.write_text("class ExecutionPlanV2: pass\n", encoding="utf-8")
    probe = tmp_path / "tests/multi_consumer.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        "\n".join(
            (
                "from market_support_crewai_agent.runtime.planning.models import ExecutionPlan",
                "def consume(left: ExecutionPlan, right: ExecutionPlan) -> None:",
                "    return None",
            )
        ),
        encoding="utf-8",
    )
    fixture = tmp_path / "two-consumers-one-target.json"
    fixture.write_text(scan_consumers(tmp_path).model_dump_json(), encoding="utf-8")

    # When: inventory validation checks target definitions separately from row counts.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--inventory",
            "--root",
            str(tmp_path),
            "--fixture",
            str(fixture),
            "--planned-fixture",
            str(ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: the checker accepts one target definition for multiple consumers.
    assert completed.returncode == 0
    assert "OK:" in completed.stdout
    assert "target_type_replacement_count_drift" not in completed.stderr


@pytest.mark.parametrize(
    ("replacement_definitions", "expected_error"),
    (
        ((), "target_type_replacement_count_drift"),
        (("first", "second"), "target_type_replacement_count_drift"),
    ),
)
def test_inventory_cli_rejects_missing_or_miscounted_target_replacement(
    tmp_path: Path,
    replacement_definitions: tuple[str, ...],
    expected_error: str,
) -> None:
    # Given: a retained consumer row whose declared replacement count is absent or duplicated.
    _write_minimal_planned_sources(tmp_path)
    probe = tmp_path / "tests/retained_consumer.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        "\n".join(
            (
                "from market_support_crewai_agent.runtime.planning.models import ExecutionPlan",
                "def consume(plan: ExecutionPlan) -> ExecutionPlan:",
                "    return plan",
            )
        ),
        encoding="utf-8",
    )
    replacement = tmp_path / "src/replacement.py"
    replacement.write_text(
        "\n".join(
            f"class ExecutionPlanV2: pass  # {name}" for name in replacement_definitions
        ),
        encoding="utf-8",
    )
    fixture = tmp_path / "target-replacement-mutation.json"
    fixture.write_text(scan_consumers(tmp_path).model_dump_json(), encoding="utf-8")

    # When: inventory validation checks raw scanner equality and target structure.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--inventory",
            "--root",
            str(tmp_path),
            "--fixture",
            str(fixture),
            "--planned-fixture",
            str(ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: the checker fails closed without printing success.
    assert completed.returncode == 1
    assert expected_error in completed.stderr
    assert "OK:" not in completed.stdout


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("path", "src/nonexistent.py"),
        ("symbol", "NonexistentBoundaryV1"),
        ("target_todo", 16),
        ("kind", "export"),
        ("expected_count", 2),
    ),
)
def test_inventory_cli_rejects_planned_ownership_structural_drift(
    tmp_path: Path,
    field: str,
    value: str | int,
) -> None:
    # Given: a schema-valid planned-symbol fixture with one structural mutation.
    source = ROOT / "tests/fixtures/planned_symbol_ownership.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["rows"][3][field] = value
    planned = tmp_path / "planned.json"
    planned.write_text(json.dumps(payload), encoding="utf-8")

    # When: the real inventory command validates the mutated planned catalog.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--inventory",
            "--planned-fixture",
            str(planned),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: path/owner/kind/count drift fails without a misleading OK marker.
    assert completed.returncode == 1
    assert "ERROR:" in completed.stderr
    assert "OK:" not in completed.stdout


def test_scanner_tracks_qualified_alias_attribute_and_dict_flow(tmp_path: Path) -> None:
    # Given: qualified legacy typing propagated through aliases, attributes, and a dict.
    legacy_module = "market_support_crewai_agent." + "schemas"
    source = tmp_path / "src/probe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            (
                f"import {legacy_module} as s",
                "RequestAlias = s.ReplyRequest",
                "def consume(request: RequestAlias):",
                "    request_alias = request",
                "    payload = {'request': request_alias}",
                "    return request_alias.conversation_key, payload",
            )
        ),
        encoding="utf-8",
    )

    # When: AST-aware migration inventory analyzes the disposable source.
    inventory = scan_consumers(tmp_path)
    kinds = {(row.symbol, row.match_kind) for row in inventory.rows}

    # Then: every qualified and propagated typed-flow boundary is owned.
    assert ("ReplyRequest", "assignment_alias") in kinds
    assert ("ReplyRequest", "parameter") in kinds
    assert ("ReplyRequest", "dict_alias") in kinds
    assert ("ReplyRequest.conversation_key", "attribute_flow") in kinds


def test_scanner_tracks_raw_identity_and_string_state_keys(tmp_path: Path) -> None:
    # Given: raw request identity reaches metadata and a state API accepts raw keys.
    legacy_module = "market_support_crewai_agent." + "schemas"
    source = tmp_path / "src/market_support_crewai_agent/runtime/state/audit.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            (
                "from typing import Any",
                f"from {legacy_module} import ReplyRequest",
                "def audit_turn(request: ReplyRequest, conversation_key: str, state_key: Any):",
                "    metadata = {'group_id': request.group_id, 'sender_id': request.sender_id}",
                "    return metadata, conversation_key, state_key",
            )
        ),
        encoding="utf-8",
    )

    # When: the migration scanner analyzes metadata and state-key signatures.
    inventory = scan_consumers(tmp_path)
    kinds = {(row.symbol, row.match_kind) for row in inventory.rows}

    # Then: raw identity and both unsafe key annotations are explicit owned rows.
    assert ("raw_legacy_identity", "metadata_flow") in kinds
    assert ("conversation_key", "unsafe_state_key_annotation") in kinds
    assert ("state_key", "unsafe_state_key_annotation") in kinds
