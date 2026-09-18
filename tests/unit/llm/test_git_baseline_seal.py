from __future__ import annotations

# pyright: reportUnusedCallResult=false

import os
import subprocess
from pathlib import Path

import pytest

from scripts.git_baseline_collector import collect_baseline_manifest
from scripts.git_baseline_seal import BaselineSealExistsError, seal_baseline


ROOT = Path(__file__).resolve().parents[3]


def _init_repository(path: Path, *, commit: bool) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    if commit:
        (path / "tracked.txt").write_bytes(b"baseline\n")
        (path / "rename.txt").write_bytes(b"rename\n")
        (path / "delete.txt").write_bytes(b"delete\n")
        subprocess.run(["git", "add", "."], cwd=path, check=True)
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
            cwd=path,
            check=True,
        )


def test_committed_manifest_seals_dirty_bytes_and_git_presence(tmp_path: Path) -> None:
    # Given: all dirty classes plus a symlink in a committed disposable repository.
    repository = tmp_path / "repository"
    _init_repository(repository, commit=True)
    (repository / "tracked.txt").write_bytes(b"staged\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    (repository / "tracked.txt").write_bytes(b"unstaged-after-index\n")
    subprocess.run(
        ["git", "mv", "rename.txt", "renamed.txt"], cwd=repository, check=True
    )
    (repository / "delete.txt").unlink()
    (repository / "untracked.bin").write_bytes(b"\x00\xffpayload")
    os.symlink("tracked.txt", repository / "link")

    # When: the complete committed baseline manifest is collected.
    manifest = collect_baseline_manifest(repository)
    by_path = {row.path: row for row in manifest.paths}

    # Then: path classes and worktree/index/HEAD metadata distinguish every byte layer.
    assert manifest.baseline_kind == "commit"
    assert manifest.head is not None
    assert "tracked.txt" in manifest.classifications.staged
    assert "tracked.txt" in manifest.classifications.unstaged
    assert "renamed.txt" in manifest.classifications.renamed
    assert "delete.txt" in manifest.classifications.deleted
    assert "untracked.bin" in manifest.classifications.untracked
    assert by_path["tracked.txt"].worktree.sha256 != by_path["tracked.txt"].index.sha256
    assert by_path["tracked.txt"].index.sha256 != by_path["tracked.txt"].head.sha256
    assert by_path["link"].worktree.file_type == "symlink"
    assert by_path["link"].worktree.sha256 is not None


def test_unborn_manifest_uses_empty_tree_without_head_blob_reads(
    tmp_path: Path,
) -> None:
    # Given: an unborn repository with staged and untracked files.
    repository = tmp_path / "unborn"
    _init_repository(repository, commit=False)
    (repository / "staged.txt").write_bytes(b"staged")
    subprocess.run(["git", "add", "staged.txt"], cwd=repository, check=True)
    (repository / "untracked.txt").write_bytes(b"untracked")
    commands: list[tuple[str, ...]] = []

    # When: collection runs with a command observer.
    manifest = collect_baseline_manifest(
        repository,
        command_observer=commands.append,
    )

    # Then: HEAD is absent, the standard empty tree is authoritative, and no HEAD blob is read.
    assert manifest.baseline_kind == "unborn"
    assert manifest.head is None
    assert manifest.empty_tree_id == "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    assert manifest.paths[0].head.present is False
    assert all(
        "HEAD" not in command
        for command in commands
        if "cat-file" in command or "ls-tree" in command
    )


def test_seal_is_canonical_exclusive_read_only_and_detects_byte_drift(
    tmp_path: Path,
) -> None:
    # Given: a committed repository and an unused evidence attempt root.
    repository = tmp_path / "repository"
    evidence_root = tmp_path / "evidence"
    _init_repository(repository, commit=True)
    (repository / "untracked.txt").write_bytes(b"before")
    before = collect_baseline_manifest(repository)

    # When: the baseline is sealed once and a byte changes afterward.
    receipt = seal_baseline(repository, evidence_root, "attempt-001")
    (repository / "untracked.txt").write_bytes(b"after")
    after = collect_baseline_manifest(repository)

    # Then: canonical bytes are digest-bound, immutable, exclusive, and byte drift is visible.
    assert (
        receipt.manifest_sha256
        == receipt.sha256_path.read_text(encoding="ascii").strip()
    )
    assert receipt.manifest_path.stat().st_mode & 0o222 == 0
    assert receipt.seal_dir.stat().st_mode & 0o222 == 0
    assert receipt.attempt_dir.stat().st_mode & 0o200
    assert before.canonical_sha256() != after.canonical_sha256()
    with pytest.raises(BaselineSealExistsError):
        seal_baseline(repository, evidence_root, "attempt-001")


def test_sensitive_paths_record_metadata_without_content_hash_or_blob(
    tmp_path: Path,
) -> None:
    # Given: an untracked secret-shaped path in an unborn repository.
    repository = tmp_path / "unborn"
    _init_repository(repository, commit=False)
    (repository / "service.secret").write_text("token-value", encoding="utf-8")

    # When: the baseline is collected and sealed.
    manifest = collect_baseline_manifest(repository)
    receipt = seal_baseline(repository, tmp_path / "evidence", "sensitive")
    row = next(item for item in manifest.paths if item.path == "service.secret")

    # Then: mode/size remain auditable while content, hashes, diffs, and blobs are excluded.
    assert row.content_policy == "excluded_sensitive"
    assert row.worktree.size == len("token-value")
    assert row.worktree.sha256 is None
    assert not any(receipt.blob_dir.iterdir())


def test_gitignore_owns_the_whole_local_planning_tree() -> None:
    # Given: the repository ignore contract after local planning state left the index.
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    # When/Then: the whole .omo tree is ignored by one rule, with no narrower duplicate.
    assert ".omo/" in lines
    assert not any(line.startswith(".omo/") and line != ".omo/" for line in lines)
