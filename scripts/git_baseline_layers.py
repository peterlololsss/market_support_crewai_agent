from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import unicodedata
from collections.abc import Callable
from pathlib import Path

from scripts.git_baseline_models import FileLayerV1


CommandObserver = Callable[[tuple[str, ...]], None]
GitEntry = tuple[str, str]


class GitBaselineError(RuntimeError):
    pass


def run_git(
    repository: Path,
    arguments: tuple[str, ...],
    *,
    observer: CommandObserver | None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    command = ("git", *arguments)
    if observer is not None:
        observer(command)
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        raise GitBaselineError(
            f"git_command_failed:{arguments[0]}:{completed.returncode}"
        )
    return completed


def normalize_path(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def is_sensitive_path(path: str) -> bool:
    lowered = path.casefold()
    name = Path(lowered).name
    parts = set(Path(lowered).parts)
    if ".omo" in parts and ("evidence" in parts or "start-work" in parts):
        return True
    if name == ".env" or name.startswith(".env."):
        return True
    if Path(name).suffix in {".key", ".pem", ".p12", ".pfx"}:
        return True
    return any(marker in name for marker in ("secret", "credential", "token"))


def worktree_layer(
    repository: Path, path: str, *, sensitive: bool
) -> tuple[FileLayerV1, bytes | None]:
    target = repository / path
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return missing_layer(), None
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        content = os.readlink(target).encode("utf-8", "surrogateescape")
        file_type = "symlink"
    elif stat.S_ISREG(metadata.st_mode):
        content = target.read_bytes()
        file_type = "regular"
    else:
        content = None
        file_type = "other"
    sha256 = (
        None if sensitive or content is None else hashlib.sha256(content).hexdigest()
    )
    return (
        FileLayerV1(
            present=True,
            file_type=file_type,
            mode=mode,
            size=metadata.st_size,
            sha256=sha256,
        ),
        None if sensitive else content,
    )


def blob_layer(
    repository: Path,
    entry: GitEntry | None,
    *,
    sensitive: bool,
    observer: CommandObserver | None,
) -> tuple[FileLayerV1, bytes | None]:
    if entry is None:
        return missing_layer(), None
    git_mode, oid = entry
    if sensitive:
        return (
            FileLayerV1(
                present=True,
                file_type="regular",
                mode=None,
                size=None,
                sha256=None,
                git_mode=git_mode,
                blob_oid=None,
            ),
            None,
        )
    content = run_git(
        repository,
        ("cat-file", "blob", oid),
        observer=observer,
    ).stdout
    return (
        FileLayerV1(
            present=True,
            file_type="symlink" if git_mode == "120000" else "regular",
            mode=None,
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            git_mode=git_mode,
            blob_oid=oid,
        ),
        content,
    )


def parse_index_entries(raw: bytes) -> dict[str, GitEntry]:
    entries: dict[str, GitEntry] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        mode, oid, stage = header.decode("ascii").split(" ")
        path = normalize_path(raw_path.decode("utf-8", "surrogateescape"))
        if stage == "0" or path not in entries:
            entries[path] = (mode, oid)
    return entries


def parse_tree_entries(raw: bytes) -> dict[str, GitEntry]:
    entries: dict[str, GitEntry] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        mode, _kind, oid = header.decode("ascii").split(" ")
        path = normalize_path(raw_path.decode("utf-8", "surrogateescape"))
        entries[path] = (mode, oid)
    return entries


def missing_layer() -> FileLayerV1:
    return FileLayerV1(
        present=False,
        file_type="missing",
        mode=None,
        size=None,
        sha256=None,
    )
