from __future__ import annotations

import hashlib
import json
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict


FileTypeV1 = Literal["missing", "regular", "symlink", "other"]
ContentPolicyV1 = Literal["hash_and_seal", "excluded_sensitive"]
BaselineKindV1 = Literal["commit", "unborn"]


class FileLayerV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    present: bool
    file_type: FileTypeV1
    mode: int | None
    size: int | None
    sha256: str | None
    git_mode: str | None = None
    blob_oid: str | None = None


class BaselinePathRecordV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    path: str
    original_path: str | None
    classifications: tuple[str, ...]
    content_policy: ContentPolicyV1
    worktree: FileLayerV1
    index: FileLayerV1
    head: FileLayerV1


class BaselineClassificationsV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    staged: tuple[str, ...]
    unstaged: tuple[str, ...]
    renamed: tuple[str, ...]
    copied: tuple[str, ...]
    deleted: tuple[str, ...]
    type_changed: tuple[str, ...]
    conflicted: tuple[str, ...]
    ignored_task: tuple[str, ...]
    untracked: tuple[str, ...]


class BaselineDiffV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    byte_size: int
    sha256: str
    sealed_name: str
    excluded_sensitive_paths: tuple[str, ...]


class GitBaselineManifestV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["git-dirty-baseline.v1"]
    baseline_kind: BaselineKindV1
    head: str | None
    symbolic_branch: str
    empty_tree_id: Literal["4b825dc642cb6eb9a060e54bf8d69288fbee4904"]
    status_sha256: str
    classifications: BaselineClassificationsV1
    staged_diff: BaselineDiffV1
    unstaged_diff: BaselineDiffV1
    paths: tuple[BaselinePathRecordV1, ...]
    exclusion_policy_id: Literal["baseline-sensitive-path-exclusion.v1"]
    exclusion_rules: tuple[str, ...]

    def canonical_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
