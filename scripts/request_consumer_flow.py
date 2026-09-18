from __future__ import annotations

import ast
from collections import Counter
from typing import Final

from scripts.request_consumer_models import ConsumerCountKey, Status

LEGACY_SYMBOL_OWNERS: Final = {
    "ReplyRequest": (2, "KernelReplyRequestV1"),
    "ActionFeedbackRequest": (3, "ActionFeedbackRequestV2"),
    "ExecutionPlan": (6, "ExecutionPlanV2"),
    "DomainContext": (6, "DomainContextV1"),
}
LEGACY_SCHEMA_MODULE: Final = "market_support_crewai_agent." + "schemas"
OWNED_IMPORT_SYMBOLS: Final = {
    LEGACY_SCHEMA_MODULE: frozenset({"ReplyRequest", "ActionFeedbackRequest"}),
    "market_support_crewai_agent.runtime.planning": frozenset({"ExecutionPlan"}),
    "market_support_crewai_agent.runtime.planning.models": frozenset({"ExecutionPlan"}),
    "market_support_crewai_agent.runtime.policy.ontology": frozenset({"DomainContext"}),
}
RAW_IDENTITY_FIELDS: Final = {
    "conversation_key",
    "group_id",
    "sender_id",
    "context_id",
}


def import_aliases(tree: ast.Module) -> tuple[set[str], dict[str, str]]:
    module_aliases: set[str] = set()
    symbol_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == LEGACY_SCHEMA_MODULE:
                    module_aliases.add(imported.asname or imported.name)
        if isinstance(node, ast.ImportFrom):
            owned_symbols = OWNED_IMPORT_SYMBOLS.get(node.module or "", frozenset())
            for imported in node.names:
                if imported.name in owned_symbols:
                    symbol_aliases[imported.asname or imported.name] = imported.name
    return module_aliases, symbol_aliases


def resolved_annotation_names(
    node: ast.expr,
    module_aliases: set[str],
    symbol_aliases: dict[str, str],
) -> set[str]:
    resolved = {
        symbol
        for item in ast.walk(node)
        if isinstance(item, ast.expr)
        if (symbol := resolve_symbol(item, module_aliases, symbol_aliases)) is not None
    }
    return resolved


def typed_variables(
    tree: ast.Module,
    module_aliases: set[str],
    symbol_aliases: dict[str, str],
) -> dict[str, str]:
    typed: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            for argument in arguments:
                if argument.annotation is not None:
                    save_single_type(
                        typed,
                        argument.arg,
                        resolved_annotation_names(
                            argument.annotation,
                            module_aliases,
                            symbol_aliases,
                        ),
                    )
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            save_single_type(
                typed,
                node.target.id,
                resolved_annotation_names(
                    node.annotation,
                    module_aliases,
                    symbol_aliases,
                ),
            )
    return typed


def propagate_aliases(
    tree: ast.Module,
    module_aliases: set[str],
    symbol_aliases: dict[str, str],
    typed: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            resolved = resolve_symbol(node.value, module_aliases, symbol_aliases)
            if resolved is not None and symbol_aliases.get(target.id) != resolved:
                symbol_aliases[target.id] = resolved
                changed = True
            if isinstance(node.value, ast.Name) and node.value.id in typed:
                legacy_type = typed[node.value.id]
                if typed.get(target.id) != legacy_type:
                    typed[target.id] = legacy_type
                    changed = True
    return symbol_aliases, typed


def resolve_symbol(
    node: ast.expr,
    module_aliases: set[str],
    symbol_aliases: dict[str, str],
) -> str | None:
    if isinstance(node, ast.Name):
        return symbol_aliases.get(node.id)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in module_aliases
        and node.attr in LEGACY_SYMBOL_OWNERS
    ):
        return node.attr
    return None


def record_symbol(
    counts: Counter[ConsumerCountKey],
    path: str,
    symbol: str,
    kind: str,
    retained: bool,
) -> None:
    owner = LEGACY_SYMBOL_OWNERS.get(symbol)
    if owner is not None:
        status: Status = "retained_boundary" if retained else "legacy"
        counts[(path, symbol, kind, owner[0], owner[1], status)] += 1


def record_unsafe_state_key(
    counts: Counter[ConsumerCountKey],
    path: str,
    parameter: str,
    annotation: ast.expr,
) -> None:
    if "/runtime/state/" not in f"/{path}" or parameter not in {
        "conversation_key",
        "state_key",
        "key",
    }:
        return
    if not annotation_names(annotation) & {"str", "Any"}:
        return
    target_todo = 3 if "issued_response" in path else 2
    counts[
        (
            path,
            parameter,
            "unsafe_state_key_annotation",
            target_todo,
            "ConversationStateKey",
            "legacy",
        )
    ] += 1


def annotation_names(node: ast.expr) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def dict_typed_symbols(node: ast.Dict, typed: dict[str, str]) -> set[str]:
    return {
        typed[value.id]
        for value in node.values
        if isinstance(value, ast.Name) and value.id in typed
    }


def dict_contains_raw_identity(node: ast.Dict, typed: dict[str, str]) -> bool:
    return any(
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id in typed
        and value.attr in RAW_IDENTITY_FIELDS
        for value in node.values
    )


def save_single_type(typed: dict[str, str], name: str, names: set[str]) -> None:
    if len(names) == 1:
        typed[name] = next(iter(names))
