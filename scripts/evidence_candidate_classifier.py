from __future__ import annotations

import hashlib
import re
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict

from scripts.git_baseline_models import GitBaselineManifestV1


CandidateCategory = Literal[
    "api_key_assignment",
    "bearer_token",
    "cn_phone",
    "email",
    "jwt",
    "private_key",
]

PATTERNS: Final[tuple[tuple[CandidateCategory, re.Pattern[bytes]], ...]] = (
    (
        "private_key",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    (
        "bearer_token",
        re.compile(rb"bearer[ \t\r\n]+[A-Za-z0-9._~+/-]{12,}={0,2}", re.I),
    ),
    (
        "api_key_assignment",
        re.compile(
            rb"(?:api[_-]?key|secret[_-]?key)[ \t]*[:=][ \t]*"
            + rb"[A-Za-z0-9._~+/-]{12,}",
            re.I,
        ),
    ),
    (
        "jwt",
        re.compile(
            rb"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\." + rb"[A-Za-z0-9_-]{8,}"
        ),
    ),
    (
        "email",
        re.compile(rb"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
    ),
    ("cn_phone", re.compile(rb"(?<![0-9])1[3-9][0-9]{9}(?![0-9])")),
)


class CandidateMetadataV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    category: CandidateCategory
    container: str
    match_length: int
    match_sha256: str
    origins: tuple[str, ...]


class CandidateReportV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["evidence-candidate-metadata.v1"]
    candidate_count: int
    candidates: tuple[CandidateMetadataV1, ...]


def classify_evidence(seal_dir: Path) -> CandidateReportV1:
    manifest = GitBaselineManifestV1.model_validate_json(
        (seal_dir / "baseline.manifest.json").read_bytes()
    )
    origins: dict[str, set[str]] = {}
    for row in manifest.paths:
        for layer in (row.head, row.index, row.worktree):
            if layer.sha256 is not None:
                origins.setdefault(layer.sha256, set()).add(row.path)

    candidates: list[CandidateMetadataV1] = []
    for blob in sorted((seal_dir / "blobs").glob("*.blob")):
        candidates.extend(
            _classify_bytes(
                blob.read_bytes(),
                container=f"blobs/{blob.name}",
                origins=tuple(sorted(origins.get(blob.stem, set()))),
            )
        )
    for diff_name in (
        manifest.staged_diff.sealed_name,
        manifest.unstaged_diff.sealed_name,
    ):
        for path, content in _framed_diff_records((seal_dir / diff_name).read_bytes()):
            candidates.extend(
                _classify_bytes(
                    content,
                    container=diff_name,
                    origins=(path,),
                )
            )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda row: (
                row.category,
                row.container,
                row.origins,
                row.match_sha256,
            ),
        )
    )
    return CandidateReportV1(
        contract_version="evidence-candidate-metadata.v1",
        candidate_count=len(ordered),
        candidates=ordered,
    )


def _classify_bytes(
    content: bytes,
    *,
    container: str,
    origins: tuple[str, ...],
) -> tuple[CandidateMetadataV1, ...]:
    rows: list[CandidateMetadataV1] = []
    for category, pattern in PATTERNS:
        for match in pattern.finditer(content):
            matched = match.group()
            rows.append(
                CandidateMetadataV1(
                    category=category,
                    container=container,
                    match_length=len(matched),
                    match_sha256=hashlib.sha256(matched).hexdigest(),
                    origins=origins,
                )
            )
    return tuple(rows)


def _framed_diff_records(content: bytes) -> tuple[tuple[str, bytes], ...]:
    rows: list[tuple[str, bytes]] = []
    offset = 0
    while offset < len(content):
        path_size = int.from_bytes(content[offset : offset + 4], "big")
        offset += 4
        path = content[offset : offset + path_size].decode("utf-8", "surrogateescape")
        offset += path_size
        diff_size = int.from_bytes(content[offset : offset + 8], "big")
        offset += 8
        rows.append((path, content[offset : offset + diff_size]))
        offset += diff_size
    return tuple(rows)


class ClassifierCliNamespace(Namespace):
    seal_dir: Path

    def __init__(self) -> None:
        super().__init__()
        self.seal_dir = Path()


def main() -> int:
    parser = ArgumentParser(description="Classify sealed evidence candidates")
    _ = parser.add_argument("seal_dir", type=Path)
    arguments = parser.parse_args(namespace=ClassifierCliNamespace())
    report = classify_evidence(arguments.seal_dir)
    _ = sys.stdout.write(report.model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
