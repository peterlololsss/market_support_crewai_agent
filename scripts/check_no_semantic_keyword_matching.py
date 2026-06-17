from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "market_support_crewai_agent"

BANNED_NAMES = {
    "LegacyHeuristic",
    "SequenceMatcher",
    "_semantic_terms",
    "_text_similarity_score",
    "_score_text_block",
    "_legacy_select_document_text",
    "_message_requests_unnamed_strategy_report",
    "_validate_sent_claims_grounded_by_ledger",
    "_validate_report_claims",
    "_unsupported_product_claim",
    "_product_like_claims",
    "explicit_send_targets",
    "detect_send_scope_conflict",
}
BANNED_FILE_NAMES = {
    "guardrail_pipeline.py",
    "send_scope_guard.py",
}
TEXTISH_NAMES = (
    "message",
    "text",
    "query",
    "normalized",
    "lowered",
    "raw",
    "line",
)
SCANNER_METHODS = {"find", "rfind", "index", "rindex"}

# These are not semantic business decisions. They are exact safety validators,
# schema/key checks, model routing, or typed-catalog candidate generation.
ALLOWED_FUNCTIONS = {
    "runtime/domain/entity_resolution.py:DefaultMentionExtractor._extract_from_text": (
        "candidate generation from explicit DomainEntity phrases"
    ),
    "runtime/domain/entity_resolution.py:DefaultCandidateGenerator.generate": (
        "candidate generation from exact IDs/names/aliases/examples only"
    ),
    "runtime/domain/entity_resolution.py:_find_phrase_occurrences": (
        "candidate span extraction against explicit typed catalog phrases"
    ),
    "runtime/domain/entity_resolution.py:_structured_unknown_strategy_mentions": (
        "candidate mention extraction; never resolves authority by itself"
    ),
    "runtime/domain/entity_resolution.py:_valid_phrase_boundary": (
        "exact token-boundary validation for candidate extraction"
    ),
    "runtime/evidence/document_mcp.py:DocumentMcpClient._post_json_rpc": (
        "exact JSON-RPC error-key validation"
    ),
    "runtime/evidence/document_mcp.py:_is_document_instruction_line": (
        "prompt-injection safety sanitizer"
    ),
    "runtime/llm/prompting/router.py:model_family_from_settings": (
        "model family routing from configured model ID"
    ),
    "runtime/validation/reply_validator.py:_first_completed_send_claim_token": (
        "pre-validation sanitizer for composer text, not allow/block authority"
    ),
    "runtime/validation/reply_validator.py:_contains_raw_locator": (
        "internal locator leak validator"
    ),
    "runtime/validation/guardrail_common.py:marker_in_trusted_document_context": (
        "exact image marker evidence validator"
    ),
}


class SemanticKeywordCheck(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.relative = path.relative_to(SRC_ROOT).as_posix()
        self.failures: list[str] = []
        self.stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in BANNED_NAMES:
            self._fail(node, f"banned matcher helper {node.name}")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if any(banned in node.name for banned in BANNED_NAMES):
            self._fail(node, f"banned matcher class {node.name}")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name in BANNED_NAMES:
                self._fail(node, f"banned import {alias.name}")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in BANNED_NAMES:
                self._fail(node, f"banned import {alias.name}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES:
            self._fail(node, f"banned matcher name {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BANNED_NAMES:
            self._fail(node, f"banned matcher attribute {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in SCANNER_METHODS
            and _name_contains_textish(_expr_name(node.func.value))
            and not self._allowed()
        ):
            self._fail(
                node,
                f"suspicious text scanner .{node.func.attr}() outside allowlist",
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.In, ast.NotIn)):
                continue
            haystack = _expr_name(comparator)
            needle = _expr_name(node.left)
            if (
                (_name_contains_textish(haystack) or needle in {"token", "keyword"})
                and not self._allowed()
            ):
                self._fail(
                    node,
                    "suspicious text membership check outside allowlist",
                )
        self.generic_visit(node)

    def _allowed(self) -> bool:
        return self._scope_key() in ALLOWED_FUNCTIONS

    def _scope_key(self) -> str:
        return f"{self.relative}:{'.'.join(self.stack)}"

    def _fail(self, node: ast.AST, reason: str) -> None:
        self.failures.append(f"{self.relative}:{node.lineno}: {reason}")


def main() -> int:
    failures: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.name in BANNED_FILE_NAMES:
            failures.append(
                f"{path.relative_to(SRC_ROOT).as_posix()}: banned compatibility file"
            )
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}: cannot parse: {exc}")
            continue
        checker = SemanticKeywordCheck(path)
        checker.visit(tree)
        failures.extend(checker.failures)

    if failures:
        for failure in failures:
            print(f"semantic-keyword-check: {failure}", file=sys.stderr)
        return 1
    print("semantic-keyword-check: ok")
    return 0


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _expr_name(node.func)
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    if isinstance(node, ast.Constant):
        return str(node.value)
    return ""


def _name_contains_textish(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in TEXTISH_NAMES)


if __name__ == "__main__":
    raise SystemExit(main())
