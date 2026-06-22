from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from market_support_crewai_agent.runtime.llm.prompting.registry import (  # noqa: E402
    PROMPT_FRAGMENT_PACKAGE,
    PROMPT_FRAGMENTS,
    PROMPT_REGISTRY,
)

PROMPT_ROOT = (
    SRC
    / Path(*PROMPT_FRAGMENT_PACKAGE.split("."))
)
DOCS_PROMPTS = ROOT / "docs" / "prompts.md"

MAX_RAW_PROMPT_CHARS = 180
PROMPT_MARKERS = (
    "You are",
    "Output only",
    "Return only",
    "Selector input JSON",
    "Verifier input JSON",
    "response schema",
    "response_format",
    "Do not call tools",
    "Do not compose",
)
BUSINESS_HIERARCHY_MARKERS = (
    "Artifact/action matrix",
    "Bank channel material rule",
    "Treat material_pack.options as a material_pack catalog",
    "Do not ask for a strategy only because",
    "Material words include",
    "Product-element words include",
    "Monthly report intent includes",
)
RAW_PROMPT_ALLOWED_FILES = {
    SRC / "market_support_crewai_agent/runtime/llm/prompting/registry.py",
    SRC / "market_support_crewai_agent/runtime/domain/capabilities/manifests.py",
    SRC / "market_support_crewai_agent/runtime/domain/compliance_policy.py",
}
RAW_PROMPT_ALLOWED_DIRS = {
    SRC / "market_support_crewai_agent/runtime/llm/prompts",
}


def main() -> int:
    failures: list[str] = []
    failures.extend(_check_documented_prompt_ids())
    failures.extend(_check_template_files())
    failures.extend(_check_raw_prompt_strings())
    failures.extend(_check_business_hierarchy_markers())
    if failures:
        for failure in failures:
            print(f"prompt-registry-check: {failure}", file=sys.stderr)
        return 1
    print("prompt-registry-check: ok")
    return 0


def _check_documented_prompt_ids() -> list[str]:
    docs = DOCS_PROMPTS.read_text(encoding="utf-8")
    failures: list[str] = []
    for prompt_id in PROMPT_REGISTRY.prompt_ids():
        if prompt_id not in docs:
            failures.append(f"prompt id is not documented: {prompt_id}")
    for agent_id in PROMPT_REGISTRY.agent_spec_ids():
        if agent_id not in docs:
            failures.append(f"agent prompt id is not documented: {agent_id}")
    return failures


def _check_template_files() -> list[str]:
    failures: list[str] = []
    registered_files = set()
    for fragment in PROMPT_FRAGMENTS:
        path = PROMPT_ROOT / fragment.template_name
        registered_files.add(path.resolve())
        if not path.exists():
            failures.append(
                f"registered prompt template is missing: {fragment.id} -> {path}"
            )

    for path in PROMPT_ROOT.rglob("*.md"):
        if path.resolve() not in registered_files:
            failures.append(f"prompt template is not registered: {path}")
    return failures


def _check_raw_prompt_strings() -> list[str]:
    failures: list[str] = []
    for path in (ROOT / "src").rglob("*.py"):
        if _is_allowed_raw_prompt_path(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"cannot parse {path}: {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            text = node.value.strip()
            if len(text) < MAX_RAW_PROMPT_CHARS:
                continue
            if any(marker in text for marker in PROMPT_MARKERS):
                failures.append(
                    f"raw prompt-like string over {MAX_RAW_PROMPT_CHARS} chars in "
                    f"{path}:{node.lineno}"
                )
    return failures


def _check_business_hierarchy_markers() -> list[str]:
    failures: list[str] = []
    for path in (ROOT / "src").rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md"}:
            continue
        if _is_allowed_raw_prompt_path(path):
            continue
        text = path.read_text(encoding="utf-8")
        for marker in BUSINESS_HIERARCHY_MARKERS:
            if marker in text:
                failures.append(
                    f"business hierarchy marker outside manifest/registry: {path}: {marker}"
                )
    return failures


def _is_allowed_raw_prompt_path(path: Path) -> bool:
    resolved = path.resolve()
    if resolved in {item.resolve() for item in RAW_PROMPT_ALLOWED_FILES}:
        return True
    return any(
        resolved == directory.resolve()
        or directory.resolve() in resolved.parents
        for directory in RAW_PROMPT_ALLOWED_DIRS
    )


if __name__ == "__main__":
    raise SystemExit(main())
