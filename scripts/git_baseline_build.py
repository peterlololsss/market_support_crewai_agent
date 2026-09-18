from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from scripts.git_baseline_layers import (
    CommandObserver,
    blob_layer,
    is_sensitive_path,
    normalize_path,
    run_git,
    worktree_layer,
)
from scripts.git_baseline_models import (
    BaselineClassificationsV1,
    BaselineDiffV1,
    BaselinePathRecordV1,
)
from scripts.request_consumer_models import GitStatusEntryV1


EMPTY_SENSITIVE_PATHS: frozenset[str] = frozenset()


def build_path_records(
    repository: Path,
    rows: tuple[GitStatusEntryV1, ...],
    path_classes: defaultdict[str, set[str]],
    index_entries: dict[str, tuple[str, str]],
    head_entries: dict[str, tuple[str, str]],
    blobs: dict[str, bytes],
    observer: CommandObserver | None,
) -> tuple[BaselinePathRecordV1, ...]:
    paths = sorted(
        {
            normalize_path(path)
            for row in rows
            for path in (row.path, row.original_path)
            if path is not None
        }
    )
    return tuple(
        _path_record(
            repository,
            path,
            rows,
            path_classes[path],
            index_entries.get(path),
            head_entries.get(path),
            blobs,
            observer,
        )
        for path in paths
    )


def classify_status(
    rows: tuple[GitStatusEntryV1, ...],
) -> tuple[BaselineClassificationsV1, defaultdict[str, set[str]]]:
    classes: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        path = normalize_path(row.path)
        if row.record_kind == "?":
            classes["untracked"].add(path)
        if row.record_kind == "!":
            classes["ignored_task"].add(path)
        if row.record_kind == "u":
            classes["conflicted"].add(path)
        if row.index_status not in {".", "?", "!"}:
            classes["staged"].add(path)
        if row.worktree_status not in {".", "?", "!"}:
            classes["unstaged"].add(path)
        for code, name in (
            ("R", "renamed"),
            ("C", "copied"),
            ("D", "deleted"),
            ("T", "type_changed"),
        ):
            if code in {row.index_status, row.worktree_status}:
                classes[name].add(path)
        if row.original_path is not None:
            classes["renamed"].add(normalize_path(row.original_path))
    by_path: defaultdict[str, set[str]] = defaultdict(set)
    for name, paths in classes.items():
        for path in paths:
            by_path[path].add(name)
    model = BaselineClassificationsV1(
        **{
            name: tuple(sorted(classes[name]))
            for name in BaselineClassificationsV1.model_fields
        }
    )
    return model, by_path


def collect_diff(
    repository: Path,
    paths: tuple[str, ...],
    *,
    baseline_kind: str,
    staged: bool,
    empty_tree_id: str,
    sensitive_paths: frozenset[str] = EMPTY_SENSITIVE_PATHS,
    observer: CommandObserver | None,
) -> tuple[bytes, tuple[str, ...]]:
    output = bytearray()
    excluded: list[str] = []
    for path in paths:
        if path in sensitive_paths or is_sensitive_path(path):
            excluded.append(path)
            continue
        arguments = ["diff", "--binary", "--no-ext-diff"]
        if staged:
            arguments.append("--cached")
            if baseline_kind == "unborn":
                arguments.append(empty_tree_id)
        arguments.extend(("--", path))
        diff = run_git(
            repository,
            tuple(arguments),
            observer=observer,
        ).stdout
        encoded_path = path.encode("utf-8")
        output.extend(len(encoded_path).to_bytes(4, "big"))
        output.extend(encoded_path)
        output.extend(len(diff).to_bytes(8, "big"))
        output.extend(diff)
    return bytes(output), tuple(sorted(excluded))


def diff_record(
    name: str,
    content: bytes,
    excluded: tuple[str, ...],
) -> BaselineDiffV1:
    return BaselineDiffV1(
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        sealed_name=name,
        excluded_sensitive_paths=excluded,
    )


def parse_porcelain_v2(status: bytes) -> tuple[GitStatusEntryV1, ...]:
    parts = status.split(b"\0")
    rows: list[GitStatusEntryV1] = []
    index = 0
    while index < len(parts) and parts[index]:
        record = parts[index].decode("utf-8", "surrogateescape")
        kind = record[0]
        if kind == "2":
            fields = record.split(" ", 9)
            rows.append(
                GitStatusEntryV1(
                    kind,
                    fields[1][0],
                    fields[1][1],
                    normalize_path(fields[9]),
                    normalize_path(parts[index + 1].decode("utf-8", "surrogateescape")),
                )
            )
            index += 2
            continue
        if kind == "u":
            fields = record.split(" ", 10)
            status_code, path = fields[1], fields[10]
        elif kind == "1":
            fields = record.split(" ", 8)
            status_code, path = fields[1], fields[8]
        else:
            status_code, path = ("??" if kind == "?" else "!!"), record[2:]
        rows.append(
            GitStatusEntryV1(
                kind,
                status_code[0],
                status_code[1],
                normalize_path(path),
                None,
            )
        )
        index += 1
    return tuple(rows)


def _path_record(
    repository: Path,
    path: str,
    rows: tuple[GitStatusEntryV1, ...],
    classifications: set[str],
    index_entry: tuple[str, str] | None,
    head_entry: tuple[str, str] | None,
    blobs: dict[str, bytes],
    observer: CommandObserver | None,
) -> BaselinePathRecordV1:
    original = next(
        (
            row.original_path
            for row in rows
            if row.path == path and row.original_path is not None
        ),
        None,
    )
    sensitive = is_sensitive_path(path) or (
        original is not None and is_sensitive_path(original)
    )
    worktree, worktree_bytes = worktree_layer(
        repository,
        path,
        sensitive=sensitive,
    )
    index, index_bytes = blob_layer(
        repository,
        index_entry,
        sensitive=sensitive,
        observer=observer,
    )
    head, head_bytes = blob_layer(
        repository,
        head_entry,
        sensitive=sensitive,
        observer=observer,
    )
    for content in (worktree_bytes, index_bytes, head_bytes):
        if content is not None:
            _ = blobs.setdefault(hashlib.sha256(content).hexdigest(), content)
    return BaselinePathRecordV1(
        path=path,
        original_path=original,
        classifications=tuple(sorted(classifications)),
        content_policy="excluded_sensitive" if sensitive else "hash_and_seal",
        worktree=worktree,
        index=index,
        head=head,
    )


def sensitive_status_paths(
    rows: tuple[GitStatusEntryV1, ...],
) -> frozenset[str]:
    sensitive: set[str] = set()
    for row in rows:
        endpoints = tuple(
            path for path in (row.path, row.original_path) if path is not None
        )
        if any(is_sensitive_path(path) for path in endpoints):
            sensitive.update(endpoints)
    return frozenset(sensitive)
