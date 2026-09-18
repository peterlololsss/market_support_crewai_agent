from __future__ import annotations

# pyright: reportAny=false
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Final

from scripts.git_baseline_models import GitBaselineManifestV1

ROOT = Path(__file__).resolve().parents[3]
PROMPT_RESOURCE_PREFIXES: Final = (
    "src/market_support_crewai_agent/runtime/prompts/",
    "tests/snapshots/prompts/",
)
EXPECTED_PROMPT_RESOURCE_PATHS: Final = frozenset(
    {
        "src/market_support_crewai_agent/runtime/prompts/fragments/guardrail/image_alignment_verifier.md",
        "tests/snapshots/prompts/alignment_verifier.txt",
        "tests/snapshots/prompts/guardrail_image_alignment_verifier.txt",
        "tests/snapshots/prompts/knowledge_composer_boundary.txt",
        "tests/snapshots/prompts/planner_intent_ds_v4pro.txt",
    }
)


def test_dirty_baseline_characterizes_current_repository_without_mutation() -> None:
    # Given: the user's current working tree before Todo 1 implementation.
    before = subprocess.run(
        ["git", "status", "--porcelain=v2", "-z", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout

    # When: the baseline is observed a second time without a write operation.
    after = subprocess.run(
        ["git", "status", "--porcelain=v2", "-z", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout

    # Then: observation is byte-stable and the known dirty tree remains dirty.
    assert before == after
    assert before


def test_authentic_pre_edit_manifest_and_protected_assets_are_sealed() -> None:
    # Given: the authentic Round-17 manifest, pointer, and protected-path comparison.
    task_root = ROOT / ".omo/evidence/scene-aware-prompt-boundaries/task-01"
    manifest_path = task_root / "round17-pre-start/baseline.manifest.json"
    pointer = json.loads(
        (
            ROOT / ".omo/start-work/evidence/task-01/authentic-baseline-pointer.json"
        ).read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (task_root / "authentic-snapshot-provenance.json").read_text(encoding="utf-8")
    )

    # When: canonical bytes and current protected files are independently hashed.
    manifest = GitBaselineManifestV1.model_validate_json(manifest_path.read_bytes())
    protected = provenance["protected_paths"]

    # Then: the authentic seal is authoritative and prompt resources still match.
    assert manifest.canonical_sha256() == pointer["authentic_manifest_sha256"]
    assert len(manifest.paths) == 504
    assert provenance["plan_sha256"] == (
        "bd47ea42216da544873f952476df96599e5710dbcc5383389a6a8e1d6eaab401"
    )
    assert provenance["sensitive_content_copied"] is False
    resources = [
        asset
        for asset in protected
        if asset["path"].startswith(PROMPT_RESOURCE_PREFIXES)
    ]
    assert {asset["path"] for asset in resources} == EXPECTED_PROMPT_RESOURCE_PATHS
    for asset in resources:
        current_sha256 = hashlib.sha256((ROOT / asset["path"]).read_bytes()).hexdigest()
        assert asset["status"] == "byte_identical"
        assert current_sha256 == asset["current_sha256"]
