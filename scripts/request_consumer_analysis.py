from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Final

from scripts.request_consumer_flow import (
    LEGACY_SYMBOL_OWNERS,
    dict_contains_raw_identity,
    dict_typed_symbols,
    import_aliases,
    propagate_aliases,
    record_symbol,
    record_unsafe_state_key,
    resolve_symbol,
    resolved_annotation_names,
    typed_variables,
)
from scripts.request_consumer_models import (
    ConsumerCountKey,
    ConsumerInventoryV1,
    ConsumerRowV1,
    SourceScanError,
    Status,
)


SUPERSEDED_INPUTS: Final = {
    "request.dist_channel_name": (5, "runtime_inputs.distribution_name"),
    "available_artifacts material_pack.options": (
        5,
        "runtime_inputs.material_pack_options",
    ),
    "plan.evidence_query": (5, "runtime_inputs.evidence_query"),
    "domain_context.strategies": (5, "runtime_inputs.strategy_ids"),
}
LOCAL_OWNED_SYMBOLS: Final = {
    "src/market_support_crewai_agent/schemas.py": frozenset(
        {"ReplyRequest", "ActionFeedbackRequest"}
    ),
    "src/market_support_crewai_agent/runtime/planning/models.py": frozenset(
        {"ExecutionPlan"}
    ),
    "src/market_support_crewai_agent/runtime/policy/ontology.py": frozenset(
        {"DomainContext"}
    ),
}


def scan_consumers(root: Path) -> ConsumerInventoryV1:
    counts: Counter[ConsumerCountKey] = Counter()
    paths = sorted((root / "src").rglob("*.py"))
    paths += sorted((root / "scripts").rglob("*.py"))
    paths += sorted((root / "tests").rglob("*.py"))
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise SourceScanError(relative, exc.lineno, exc.msg) from exc
        except UnicodeDecodeError as exc:
            raise SourceScanError(relative, None, exc.reason) from exc
        _scan_tree(counts, relative, tree)
    rows = [
        ConsumerRowV1(
            path=key[0],
            symbol=key[1],
            match_kind=key[2],
            expected_count=count,
            target_todo=key[3],
            target_type=key[4],
            status=key[5],
        )
        for key, count in sorted(counts.items())
    ]
    return ConsumerInventoryV1(
        contract_version="request-consumer-migration.v1",
        rows=tuple(rows),
    )


def _scan_tree(
    counts: Counter[ConsumerCountKey],
    relative: str,
    tree: ast.Module,
) -> None:
    retained = relative.startswith("tests/") or relative.endswith("schemas.py")
    module_aliases, symbol_aliases = import_aliases(tree)
    symbol_aliases.update(
        {symbol: symbol for symbol in LOCAL_OWNED_SYMBOLS.get(relative, frozenset())}
    )
    symbol_aliases, _ = propagate_aliases(tree, module_aliases, symbol_aliases, {})
    typed = typed_variables(tree, module_aliases, symbol_aliases)
    symbol_aliases, typed = propagate_aliases(
        tree,
        module_aliases,
        symbol_aliases,
        typed,
    )
    for node in ast.walk(tree):
        _scan_node(
            counts,
            relative,
            node,
            retained,
            module_aliases,
            symbol_aliases,
            typed,
        )


def _scan_node(
    counts: Counter[ConsumerCountKey],
    path: str,
    node: ast.AST,
    retained: bool,
    module_aliases: set[str],
    symbol_aliases: dict[str, str],
    typed: dict[str, str],
) -> None:
    if isinstance(node, ast.ImportFrom):
        for imported in node.names:
            alias = imported.asname or imported.name
            resolved = symbol_aliases.get(alias)
            if resolved is not None and resolved == imported.name:
                record_symbol(counts, path, resolved, "import", retained)
    if isinstance(node, ast.Call):
        resolved = resolve_symbol(node.func, module_aliases, symbol_aliases)
        if resolved is not None:
            record_symbol(counts, path, resolved, "construction", retained)
    if isinstance(node, ast.AnnAssign):
        for name in resolved_annotation_names(
            node.annotation,
            module_aliases,
            symbol_aliases,
        ):
            record_symbol(counts, path, name, "annotation", retained)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        _scan_function(
            counts,
            path,
            node,
            retained,
            module_aliases,
            symbol_aliases,
        )
    if isinstance(node, ast.Assign):
        _scan_assignment(
            counts,
            path,
            node,
            retained,
            module_aliases,
            symbol_aliases,
            typed,
        )
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        legacy_type = typed.get(node.value.id)
        if legacy_type is not None and node.attr == "conversation_key":
            attribute_status: Status = "retained_boundary" if retained else "legacy"
            counts[
                (
                    path,
                    f"{legacy_type}.conversation_key",
                    "attribute_flow",
                    2,
                    "ConversationStateKey",
                    attribute_status,
                )
            ] += 1
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        owner = SUPERSEDED_INPUTS.get(node.value)
        if owner is not None and path != "scripts/request_consumer_analysis.py":
            literal_status: Status = "retained_boundary" if retained else "legacy"
            counts[
                (path, node.value, "literal", owner[0], owner[1], literal_status)
            ] += 1


def _scan_function(
    counts: Counter[ConsumerCountKey],
    path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    retained: bool,
    module_aliases: set[str],
    symbol_aliases: dict[str, str],
) -> None:
    arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    for argument in arguments:
        if argument.annotation is None:
            continue
        for name in resolved_annotation_names(
            argument.annotation,
            module_aliases,
            symbol_aliases,
        ):
            record_symbol(counts, path, name, "parameter", retained)
        record_unsafe_state_key(counts, path, argument.arg, argument.annotation)
    if node.returns is not None:
        for name in resolved_annotation_names(
            node.returns,
            module_aliases,
            symbol_aliases,
        ):
            record_symbol(counts, path, name, "return", retained)


def _scan_assignment(
    counts: Counter[ConsumerCountKey],
    path: str,
    node: ast.Assign,
    retained: bool,
    module_aliases: set[str],
    symbol_aliases: dict[str, str],
    typed: dict[str, str],
) -> None:
    resolved = resolve_symbol(node.value, module_aliases, symbol_aliases)
    if resolved in LEGACY_SYMBOL_OWNERS:
        record_symbol(counts, path, resolved, "assignment_alias", retained)
    if not isinstance(node.value, ast.Dict):
        return
    for symbol in dict_typed_symbols(node.value, typed):
        record_symbol(counts, path, symbol, "dict_alias", retained)
    if dict_contains_raw_identity(node.value, typed):
        counts[
            (
                path,
                "raw_legacy_identity",
                "metadata_flow",
                4,
                "ConversationStateKey",
                "legacy",
            )
        ] += 1
