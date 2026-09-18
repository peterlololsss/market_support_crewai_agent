from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scripts.git_baseline_build import (
    build_path_records,
    classify_status,
    collect_diff,
    diff_record,
    parse_porcelain_v2,
    sensitive_status_paths,
)
from scripts.git_baseline_layers import (
    CommandObserver,
    GitBaselineError,
    parse_index_entries,
    parse_tree_entries,
    run_git,
)
from scripts.git_baseline_models import BaselineKindV1, GitBaselineManifestV1
from scripts.request_consumer_models import GitBaselineV1


EMPTY_TREE_ID: Final = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
EXCLUSION_RULES: Final = (
    "exclude .omo evidence/start-work paths from content hashing and blob copies",
    "exclude .env and .env.* paths",
    "exclude key/pem/p12/pfx suffixes",
    "exclude basenames containing secret, credential, or token",
)


@dataclass(frozen=True, slots=True)
class BaselineCollectionV1:
    manifest: GitBaselineManifestV1
    staged_diff: bytes
    unstaged_diff: bytes
    blobs: dict[str, bytes]


def collect_git_baseline(repository: Path) -> GitBaselineV1:
    head_result = run_git(
        repository,
        ("rev-parse", "--verify", "HEAD"),
        observer=None,
        check=False,
    )
    status = run_git(
        repository,
        ("status", "--porcelain=v2", "-z", "--untracked-files=all"),
        observer=None,
    ).stdout
    head = (
        head_result.stdout.decode("ascii").strip()
        if head_result.returncode == 0
        else None
    )
    return GitBaselineV1(head=head, status=status, rows=parse_porcelain_v2(status))


def collect_baseline_manifest(
    repository: Path,
    *,
    command_observer: CommandObserver | None = None,
) -> GitBaselineManifestV1:
    return collect_baseline_bundle(
        repository,
        command_observer=command_observer,
    ).manifest


def collect_baseline_bundle(
    repository: Path,
    *,
    command_observer: CommandObserver | None = None,
) -> BaselineCollectionV1:
    _verify_repository(repository, command_observer)
    head, baseline_kind = _resolve_head(repository, command_observer)
    branch = _symbolic_branch(repository, command_observer)
    status = run_git(
        repository,
        (
            "status",
            "--porcelain=v2",
            "-z",
            "--untracked-files=all",
            "--ignored=matching",
        ),
        observer=command_observer,
    ).stdout
    rows = tuple(
        row
        for row in parse_porcelain_v2(status)
        if row.record_kind != "!"
        or row.path.startswith(".omo/evidence/scene-aware-prompt-boundaries/")
    )
    index_entries = parse_index_entries(
        run_git(
            repository,
            ("ls-files", "-s", "-z"),
            observer=command_observer,
        ).stdout
    )
    head_entries = _head_entries(
        repository,
        baseline_kind,
        command_observer,
    )
    classifications, path_classes = classify_status(rows)
    sensitive_paths = sensitive_status_paths(rows)
    blobs: dict[str, bytes] = {}
    records = build_path_records(
        repository,
        rows,
        path_classes,
        index_entries,
        head_entries,
        blobs,
        command_observer,
    )
    staged_bytes, staged_excluded = collect_diff(
        repository,
        classifications.staged,
        baseline_kind=baseline_kind,
        staged=True,
        empty_tree_id=EMPTY_TREE_ID,
        sensitive_paths=sensitive_paths,
        observer=command_observer,
    )
    unstaged_bytes, unstaged_excluded = collect_diff(
        repository,
        classifications.unstaged,
        baseline_kind=baseline_kind,
        staged=False,
        empty_tree_id=EMPTY_TREE_ID,
        sensitive_paths=sensitive_paths,
        observer=command_observer,
    )
    manifest = GitBaselineManifestV1(
        contract_version="git-dirty-baseline.v1",
        baseline_kind=baseline_kind,
        head=head,
        symbolic_branch=branch,
        empty_tree_id=EMPTY_TREE_ID,
        status_sha256=hashlib.sha256(status).hexdigest(),
        classifications=classifications,
        staged_diff=diff_record("staged.diff", staged_bytes, staged_excluded),
        unstaged_diff=diff_record("unstaged.diff", unstaged_bytes, unstaged_excluded),
        paths=records,
        exclusion_policy_id="baseline-sensitive-path-exclusion.v1",
        exclusion_rules=EXCLUSION_RULES,
    )
    return BaselineCollectionV1(manifest, staged_bytes, unstaged_bytes, blobs)


def _verify_repository(repository: Path, observer: CommandObserver | None) -> None:
    git_dir = run_git(repository, ("rev-parse", "--git-dir"), observer=observer)
    inside = run_git(
        repository,
        ("rev-parse", "--is-inside-work-tree"),
        observer=observer,
    )
    if not git_dir.stdout.strip() or inside.stdout.strip() != b"true":
        raise GitBaselineError("not_a_git_work_tree")


def _resolve_head(
    repository: Path,
    observer: CommandObserver | None,
) -> tuple[str | None, BaselineKindV1]:
    result = run_git(
        repository,
        ("rev-parse", "--verify", "HEAD"),
        observer=observer,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.decode("ascii").strip(), "commit"
    symbolic = run_git(
        repository,
        ("symbolic-ref", "--quiet", "HEAD"),
        observer=observer,
        check=False,
    )
    revisions = run_git(
        repository,
        ("rev-list", "--all", "--max-count=1"),
        observer=observer,
        check=False,
    )
    if symbolic.returncode == 0 and not revisions.stdout.strip():
        return None, "unborn"
    raise GitBaselineError("head_verification_failed")


def _symbolic_branch(repository: Path, observer: CommandObserver | None) -> str:
    result = run_git(
        repository,
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        observer=observer,
        check=False,
    )
    return (
        result.stdout.decode("utf-8").strip()
        if result.returncode == 0
        else "(detached)"
    )


def _head_entries(
    repository: Path,
    baseline_kind: str,
    observer: CommandObserver | None,
) -> dict[str, tuple[str, str]]:
    if baseline_kind == "unborn":
        return {}
    return parse_tree_entries(
        run_git(
            repository,
            ("ls-tree", "-rz", "HEAD"),
            observer=observer,
        ).stdout
    )
