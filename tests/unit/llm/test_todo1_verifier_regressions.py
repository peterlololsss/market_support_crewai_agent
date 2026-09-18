from __future__ import annotations

# pyright: reportAny=false, reportUnusedCallResult=false
import json
import subprocess
from pathlib import Path

from scripts.git_baseline_build import parse_porcelain_v2
from scripts.git_baseline_collector import collect_baseline_manifest
from scripts.git_baseline_seal import seal_baseline
from scripts.request_consumer_analysis import scan_consumers

ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER = ROOT / "scripts/evidence_candidate_classifier.py"


def _init_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def test_unmerged_porcelain_record_parses_xy_and_path() -> None:
    # Given: Git porcelain-v2's fixed-width unmerged record shape.
    raw = (
        b"u UU N... 100644 100644 100644 100644 "
        + b"1" * 40
        + b" "
        + b"2" * 40
        + b" "
        + b"3" * 40
        + b" conflict.txt\0"
    )

    # When: the status boundary parses the record.
    rows = parse_porcelain_v2(raw)

    # Then: the conflict statuses and path exclude porcelain metadata.
    assert rows[0].record_kind == "u"
    assert rows[0].index_status == "U"
    assert rows[0].worktree_status == "U"
    assert rows[0].path == "conflict.txt"


def test_scanner_ignores_unrelated_legacy_symbol_names(tmp_path: Path) -> None:
    # Given: unrelated modules and attributes reuse legacy class names.
    source = tmp_path / "src/unrelated.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            (
                "from unrelated_package import ReplyRequest",
                "import unrelated_package as other",
                "def consume(request: ReplyRequest):",
                "    return other.ReplyRequest(request)",
            )
        ),
        encoding="utf-8",
    )

    # When: the migration scanner analyzes the source.
    inventory = scan_consumers(tmp_path)

    # Then: names without an owned source module create no migration rows.
    assert inventory.rows == ()


def test_dict_alias_keeps_action_feedback_request_ownership(tmp_path: Path) -> None:
    # Given: an ActionFeedbackRequest flows into a dictionary alias.
    legacy_module = "market_support_crewai_agent." + "schemas"
    source = tmp_path / "src/feedback.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            (
                f"from {legacy_module} import ActionFeedbackRequest",
                "def consume(request: ActionFeedbackRequest):",
                "    payload = {'request': request}",
                "    return payload",
            )
        ),
        encoding="utf-8",
    )

    # When: the migration scanner analyzes the typed flow.
    inventory = scan_consumers(tmp_path)
    kinds = {(row.symbol, row.match_kind, row.target_todo) for row in inventory.rows}

    # Then: the dictionary remains owned by feedback migration, never ReplyRequest.
    assert ("ActionFeedbackRequest", "dict_alias", 3) in kinds
    assert not any(row.symbol == "ReplyRequest" for row in inventory.rows)


def test_sensitive_rename_excludes_source_and_destination_bytes(tmp_path: Path) -> None:
    # Given: a sensitive committed source is renamed to an innocuous destination.
    repository = tmp_path / "repository"
    _init_repository(repository)
    (repository / "service.secret").write_bytes(b"sensitive-placeholder")
    subprocess.run(["git", "add", "service.secret"], cwd=repository, check=True)
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
    subprocess.run(
        ["git", "mv", "service.secret", "public.txt"],
        cwd=repository,
        check=True,
    )

    # When: the renamed repository is collected.
    manifest = collect_baseline_manifest(repository)
    by_path = {row.path: row for row in manifest.paths}

    # Then: both rename endpoints and their diff remain metadata-only.
    assert by_path["public.txt"].content_policy == "excluded_sensitive"
    assert by_path["public.txt"].worktree.sha256 is None
    assert "public.txt" in manifest.staged_diff.excluded_sensitive_paths


def test_attempt_keeps_writable_artifacts_outside_immutable_seal(
    tmp_path: Path,
) -> None:
    # Given: a disposable repository and plan-named attempt artifact.
    repository = tmp_path / "repository"
    _init_repository(repository)
    (repository / "tracked.txt").write_bytes(b"baseline")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)

    # When: the baseline is sealed and a sibling artifact is written.
    receipt = seal_baseline(repository, tmp_path / "evidence", "attempt-001")
    artifact = receipt.attempt_dir / "task-01-prompt-budgets.json"
    artifact.write_text("{}\n", encoding="utf-8")

    # Then: only seal bytes are immutable while attempt evidence remains writable.
    assert receipt.seal_dir.stat().st_mode & 0o222 == 0
    assert receipt.attempt_dir.stat().st_mode & 0o200
    assert artifact.read_text(encoding="utf-8") == "{}\n"


def test_candidate_classifier_emits_metadata_only(tmp_path: Path) -> None:
    # Given: sealed evidence containing a non-secret bearer-shaped test fixture.
    repository = tmp_path / "repository"
    _init_repository(repository)
    (repository / "fixture.txt").write_text(
        "Authorization: Bearer example-placeholder-token",
        encoding="utf-8",
    )
    receipt = seal_baseline(repository, tmp_path / "evidence", "attempt-001")

    # When: the replayable classifier scans the immutable seal.
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.evidence_candidate_classifier",
            str(receipt.seal_dir),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    # Then: output exposes metadata fields only, never matched bytes.
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["candidate_count"] == 1
    assert set(report["candidates"][0]) == {
        "category",
        "container",
        "match_length",
        "match_sha256",
        "origins",
    }
    assert "example-placeholder-token" not in completed.stdout
