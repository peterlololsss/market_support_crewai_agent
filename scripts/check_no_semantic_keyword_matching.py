from __future__ import annotations

import ast
import sys
from pathlib import Path

from typing_extensions import override

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
COLLECTION_TYPE_NAMES = {"dict", "frozenset", "list", "set", "tuple"}
CANONICAL_KEY_LITERALS = {"error"}
GUARD_ITERABLE_NAMES = {
    "_COMPLETED_SEND_CLAIM_TOKENS",
    "_RAW_LOCATOR_TOKENS",
    "forbidden_fields",
}


class SemanticKeywordCheck(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.relative: str = path.relative_to(SRC_ROOT).as_posix()
        self.failures: list[str] = []
        self.stack: list[str] = []
        self.bindings: list[dict[str, str]] = [{}]

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name in BANNED_NAMES:
            self._fail(node, f"banned matcher helper {node.name}")
        self.stack.append(node.name)
        arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        self.bindings.append(
            {
                argument.arg: _annotation_role(argument.annotation)
                or {"model_name": "model", "key": "key"}.get(argument.arg, "plain")
                for argument in arguments
            }
        )
        self.generic_visit(node)
        _ = self.bindings.pop()
        _ = self.stack.pop()

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if any(banned in node.name for banned in BANNED_NAMES):
            self._fail(node, f"banned matcher class {node.name}")
        self.stack.append(node.name)
        self.bindings.append({})
        self.generic_visit(node)
        _ = self.bindings.pop()
        _ = self.stack.pop()

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        role = self._expr_role(node.value)
        for target in node.targets:
            self._bind(target, role)
        self.generic_visit(node)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        role = _annotation_role(node.annotation) or self._expr_role(node.value)
        self._bind(node.target, role)
        self.generic_visit(node)

    @override
    def visit_For(self, node: ast.For) -> None:
        self._bind_iteration(node.target, node.iter)
        self.generic_visit(node)

    @override
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.bindings.append({})
        for generator in node.generators:
            self.visit(generator.iter)
            self._bind_iteration(generator.target, generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        self.visit(node.elt)
        _ = self.bindings.pop()

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name in BANNED_NAMES:
                self._fail(node, f"banned import {alias.name}")
        self.generic_visit(node)

    @override
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in BANNED_NAMES:
                self._fail(node, f"banned import {alias.name}")
        self.generic_visit(node)

    @override
    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES:
            self._fail(node, f"banned matcher name {node.id}")

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BANNED_NAMES:
            self._fail(node, f"banned matcher attribute {node.attr}")
        self.generic_visit(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in SCANNER_METHODS
            and _name_contains_textish(_expr_name(node.func.value))
            and not (node.args and _is_syntax_literal(node.args[0]))
        ):
            self._fail(
                node,
                f"suspicious text scanner .{node.func.attr}() outside allowlist",
            )
        self.generic_visit(node)

    @override
    def visit_Compare(self, node: ast.Compare) -> None:
        needle_node = node.left
        for op, comparator in zip(node.ops, node.comparators):
            haystack = _expr_name(comparator)
            needle = _expr_name(needle_node)
            suspicious = isinstance(op, (ast.In, ast.NotIn)) and (
                _name_contains_textish(haystack) or needle in {"token", "keyword"}
            )
            needle_role = self._expr_role(needle_node)
            comparator_role = self._expr_role(comparator)
            canonical_key = (
                needle_role == "key"
                or _is_canonical_name(needle_node)
                or isinstance(needle_node, ast.Constant)
                and needle_node.value in CANONICAL_KEY_LITERALS
            )
            safe_operand_membership = (
                comparator_role == "mapping"
                and canonical_key
                or comparator_role in {"key", "model"}
                or needle_role == "guard"
                or comparator_role == "collection"
                and (needle_role == "key" or _is_canonical_name(needle_node))
            )
            if suspicious and not (
                _is_syntax_literal(needle_node) or safe_operand_membership
            ):
                self._fail(
                    node,
                    "suspicious text membership check outside allowlist",
                )
            needle_node = comparator
        self.generic_visit(node)

    def _bind(self, target: ast.expr, role: str) -> None:
        if isinstance(target, ast.Name):
            self.bindings[-1][target.id] = role

    def _bind_iteration(self, target: ast.expr, iterator: ast.expr) -> None:
        if isinstance(target, (ast.List, ast.Tuple)):
            for element in target.elts:
                self._bind_iteration(element, iterator)
            return
        if isinstance(target, ast.Name):
            iterator_name = _expr_name(iterator).rsplit(".", maxsplit=1)[-1]
            role = "guard" if iterator_name in GUARD_ITERABLE_NAMES else "plain"
            if (
                role == "plain"
                and target.id == "key"
                and _iterates_parsed_keys(iterator)
            ):
                role = "key"
            self._bind(target, role)

    def _expr_role(self, node: ast.AST | None) -> str:
        if isinstance(node, ast.Name):
            for scope in reversed(self.bindings):
                if node.id in scope:
                    return scope[node.id]
            return "plain"
        if isinstance(node, (ast.Dict, ast.DictComp)):
            return "mapping"
        if isinstance(node, (ast.List, ast.ListComp, ast.Set, ast.SetComp, ast.Tuple)):
            return "collection"
        if isinstance(node, ast.Call):
            called = _expr_name(node.func).rsplit(".", maxsplit=1)[-1]
            if called in {"parse_mcp_message", "dict"}:
                return "mapping"
            if called in COLLECTION_TYPE_NAMES:
                return "collection"
            if isinstance(node.func, ast.Attribute) and called in {
                "casefold",
                "lower",
                "strip",
            }:
                return self._expr_role(node.func.value)
        return "plain"

    def _fail(self, node: ast.AST, reason: str) -> None:
        line_no = node.lineno if isinstance(node, (ast.expr, ast.stmt)) else 0
        self.failures.append(f"{self.relative}:{line_no}: {reason}")


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


def _is_syntax_literal(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and bool(node.value)
        and not any(character.isalnum() for character in node.value)
    )


def _is_canonical_name(node: ast.AST) -> bool:
    return _expr_name(node).rsplit(".", maxsplit=1)[-1].startswith("canonical_")


def _annotation_role(node: ast.AST | None) -> str:
    if node is None:
        return ""
    base = node.value if isinstance(node, ast.Subscript) else node
    name = _expr_name(base).rsplit(".", maxsplit=1)[-1]
    if name == "dict":
        return "mapping"
    return "collection" if name in COLLECTION_TYPE_NAMES else ""


def _iterates_parsed_keys(node: ast.AST) -> bool:
    return (
        _expr_name(node).endswith("parse_qsl")
        or isinstance(node, ast.BinOp)
        and (_iterates_parsed_keys(node.left) or _iterates_parsed_keys(node.right))
    )


if __name__ == "__main__":
    raise SystemExit(main())
