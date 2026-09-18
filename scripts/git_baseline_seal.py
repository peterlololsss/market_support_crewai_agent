from __future__ import annotations

import hashlib
import json
import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

from scripts.git_baseline_collector import collect_baseline_bundle


class BaselineSealExistsError(FileExistsError):
    pass


@dataclass(frozen=True, slots=True)
class BaselineSealReceiptV1:
    attempt_dir: Path
    seal_dir: Path
    manifest_path: Path
    sha256_path: Path
    transcript_path: Path
    blob_dir: Path
    manifest_sha256: str


def seal_baseline(
    repository: Path,
    evidence_root: Path,
    run_id: str,
) -> BaselineSealReceiptV1:
    evidence_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = evidence_root / run_id
    try:
        attempt_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise BaselineSealExistsError(f"baseline_attempt_exists:{run_id}") from exc
    seal_dir = attempt_dir / "seal"
    seal_dir.mkdir(mode=0o700)
    blob_dir = seal_dir / "blobs"
    blob_dir.mkdir(mode=0o700)
    try:
        collection = collect_baseline_bundle(repository)
        manifest_bytes = collection.manifest.canonical_bytes()
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path = seal_dir / "baseline.manifest.json"
        sha256_path = seal_dir / "baseline.sha256"
        transcript_path = seal_dir / "task-01-baseline.txt"
        _write_exclusive(manifest_path, manifest_bytes)
        _write_exclusive(sha256_path, (digest + "\n").encode("ascii"))
        _write_exclusive(
            seal_dir / collection.manifest.staged_diff.sealed_name,
            collection.staged_diff,
        )
        _write_exclusive(
            seal_dir / collection.manifest.unstaged_diff.sealed_name,
            collection.unstaged_diff,
        )
        for blob_hash, content in sorted(collection.blobs.items()):
            _write_exclusive(blob_dir / f"{blob_hash}.blob", content)
        transcript = (
            f"baseline_kind={collection.manifest.baseline_kind}\n"
            f"head={collection.manifest.head}\n"
            f"branch={collection.manifest.symbolic_branch}\n"
            f"manifest_sha256={digest}\n"
            f"path_count={len(collection.manifest.paths)}\n"
            "content_policy=baseline-sensitive-path-exclusion.v1\n"
        ).encode("utf-8")
        _write_exclusive(transcript_path, transcript)
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != digest:
            raise OSError("sealed_manifest_digest_mismatch")
        _make_read_only(seal_dir)
        return BaselineSealReceiptV1(
            attempt_dir=attempt_dir,
            seal_dir=seal_dir,
            manifest_path=manifest_path,
            sha256_path=sha256_path,
            transcript_path=transcript_path,
            blob_dir=blob_dir,
            manifest_sha256=digest,
        )
    except OSError:
        raise


def _write_exclusive(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        _ = stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)


class SealCliNamespace(Namespace):
    repository: Path
    evidence_root: Path
    run_id: str

    def __init__(self) -> None:
        super().__init__()
        self.repository = Path()
        self.evidence_root = Path()
        self.run_id = ""


def main() -> int:
    parser = ArgumentParser(description="Seal a deterministic Git baseline")
    _ = parser.add_argument("repository", type=Path)
    _ = parser.add_argument("evidence_root", type=Path)
    _ = parser.add_argument("run_id")
    arguments = parser.parse_args(namespace=SealCliNamespace())
    receipt = seal_baseline(
        arguments.repository,
        arguments.evidence_root,
        arguments.run_id,
    )
    _ = sys.stdout.write(
        json.dumps(
            {
                "attempt_dir": str(receipt.attempt_dir),
                "manifest_sha256": receipt.manifest_sha256,
                "seal_dir": str(receipt.seal_dir),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
