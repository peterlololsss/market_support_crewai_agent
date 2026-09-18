from __future__ import annotations

# noqa: SIZE_OK

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Final

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.git_baseline_collector import collect_git_baseline
from scripts.request_consumer_analysis import scan_consumers
from scripts.request_consumer_models import (
    ConsumerInventoryV1,
    ConsumerRowV1,
    InventoryContractError,
    PlannedSymbolInventoryV1,
    PlannedSymbolRowV1,
    SourceScanError,
)


ROOT: Final = Path(__file__).resolve().parents[1]
PLANNED_SYMBOL_STRUCTURAL_SHA256: Final = (
    "82554239fe2d05b2cc191ff15b20a5bd4cceb3dd1a704ac8ba78ae78802292dc"
)
PLANNED_SYMBOL_TODO_1_14_PREFIX_SHA256: Final = (
    "d5bb481164645b1feeb6a9fb13d10ae3e856787e124ddc868729322dfa49928c"
)
RECALL_MODELS_MODULE: Final = (
    "market_support_crewai_agent.runtime.recall.question_models"
)
RECALL_COMPATIBILITY_PATH: Final = (
    "src/market_support_crewai_agent/runtime/recall/question_recall_models.py"
)
RECALL_MODEL_SYMBOLS: Final = (
    "QuestionRecallDecision",
    "QuestionRecallStatus",
    "QuestionRecallSource",
    "QuestionRecallEntry",
    "QuestionRecallCandidate",
    "MatchDraft",
    "QuestionRecallMatch",
    "make_question_recall_match",
    "execution_plan_for_question_recall",
    "candidate_ids",
)
PENDING_MIGRATION_STATUSES: Final = ("legacy", "planned_boundary")


def _load_inventory(path: Path) -> ConsumerInventoryV1:
    return ConsumerInventoryV1.model_validate_json(path.read_bytes())


def _load_planned(path: Path) -> PlannedSymbolInventoryV1:
    return PlannedSymbolInventoryV1.model_validate_json(path.read_bytes())


def validate_inventory(
    root: Path,
    fixture: Path,
    planned_fixture: Path,
    phase: int | None,
    complete: bool,
) -> None:
    actual = scan_consumers(root)
    actual = _with_recall_migration_rows(root, actual)
    expected = _load_inventory(fixture)
    if actual != expected:
        raise InventoryContractError("request_consumer_inventory_drift")
    planned = _load_planned(planned_fixture)
    source_symbol_counts = _source_symbol_counts(root)
    target_counts = _target_definition_expectations(root, planned)
    for row in expected.rows:
        if row.status not in {"migrated", "retained_boundary"}:
            continue
        expected_target_count = target_counts.get(row.target_type, 1)
        actual_target_count = source_symbol_counts[row.target_type]
        if actual_target_count != expected_target_count:
            raise InventoryContractError("target_type_replacement_count_drift")
    structural_rows = [
        row.model_dump(
            mode="json",
            include={"symbol", "path", "target_todo", "kind", "expected_count"},
        )
        for row in planned.rows
    ]
    structural_digest = hashlib.sha256(
        json.dumps(
            structural_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if structural_digest != PLANNED_SYMBOL_STRUCTURAL_SHA256:
        raise InventoryContractError("planned_symbol_ownership_drift")
    prefix_rows = [row for row in planned.rows if row.target_todo <= 14]
    prefix_structural_rows = [
        row.model_dump(
            mode="json",
            include={"symbol", "path", "target_todo", "kind", "expected_count"},
        )
        for row in prefix_rows
    ]
    prefix_digest = hashlib.sha256(
        json.dumps(
            prefix_structural_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if prefix_digest != PLANNED_SYMBOL_TODO_1_14_PREFIX_SHA256:
        raise InventoryContractError("planned_symbol_ownership_drift")
    for row in planned.rows:
        if row.status in {"migrated", "retained_boundary"}:
            if _count_planned_symbol(root, row) != row.expected_count:
                raise InventoryContractError("planned_symbol_count_drift")
    if phase is not None:
        phase_error = _raw_phase_pending_reason(expected.rows, planned.rows, phase)
        if phase_error is not None:
            raise InventoryContractError(phase_error)
    if complete:
        completeness_error = _raw_completeness_pending_reason(
            expected.rows,
            planned.rows,
        )
        if completeness_error is not None:
            raise InventoryContractError(completeness_error)


def _is_pending_migration_status(status: str) -> bool:
    return status in PENDING_MIGRATION_STATUSES


def _raw_completeness_pending_reason(
    consumer_rows: tuple[ConsumerRowV1, ...],
    planned_rows: tuple[PlannedSymbolRowV1, ...],
) -> str | None:
    for row in consumer_rows:
        if _is_pending_migration_status(row.status):
            return (
                f"migration_inventory_incomplete:consumer_row:{row.symbol}:{row.status}"
            )
    for row in planned_rows:
        if _is_pending_migration_status(row.status):
            return f"migration_inventory_incomplete:planned_symbol:{row.symbol}:{row.status}"
    return None


def _raw_phase_pending_reason(
    consumer_rows: tuple[ConsumerRowV1, ...],
    planned_rows: tuple[PlannedSymbolRowV1, ...],
    phase: int,
) -> str | None:
    for row in consumer_rows:
        if row.target_todo <= phase and _is_pending_migration_status(row.status):
            return f"phase_has_pending_consumer_rows:consumer_row:{row.symbol}:{row.status}"
    for row in planned_rows:
        if row.target_todo <= phase and _is_pending_migration_status(row.status):
            return f"phase_has_pending_consumer_rows:planned_symbol:{row.symbol}:{row.status}"
    return None


def _with_recall_migration_rows(
    root: Path,
    inventory: ConsumerInventoryV1,
) -> ConsumerInventoryV1:
    existing_keys = {(row.path, row.symbol, row.match_kind) for row in inventory.rows}
    recall_rows = [
        row
        for row in _scan_recall_migration_rows(root)
        if (row.path, row.symbol, row.match_kind) not in existing_keys
    ]
    return ConsumerInventoryV1(
        contract_version=inventory.contract_version,
        rows=(*inventory.rows, *recall_rows),
    )


def _scan_recall_migration_rows(root: Path) -> tuple[ConsumerRowV1, ...]:
    rows: list[ConsumerRowV1] = []
    for path in sorted((root / "src").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise SourceScanError(relative, exc.lineno, exc.msg) from exc
        except UnicodeDecodeError as exc:
            raise SourceScanError(relative, None, exc.reason) from exc
        retained = relative == RECALL_COMPATIBILITY_PATH
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.ImportFrom)
                or node.module != RECALL_MODELS_MODULE
            ):
                continue
            for imported in node.names:
                if imported.name not in RECALL_MODEL_SYMBOLS:
                    continue
                rows.append(
                    ConsumerRowV1(
                        path=relative,
                        symbol=imported.name,
                        match_kind="import",
                        expected_count=1,
                        target_todo=8,
                        target_type=imported.name,
                        status="retained_boundary" if retained else "migrated",
                    )
                )
        if not retained:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                continue
            exported = {
                item.value
                for item in node.value.elts
                if isinstance(item, ast.Constant)
                and isinstance(item.value, str)
                and item.value in RECALL_MODEL_SYMBOLS
            }
            rows.extend(
                ConsumerRowV1(
                    path=relative,
                    symbol=symbol,
                    match_kind="export",
                    expected_count=1,
                    target_todo=8,
                    target_type=symbol,
                    status="retained_boundary",
                )
                for symbol in sorted(exported)
            )
    return tuple(sorted(rows, key=lambda row: (row.path, row.symbol, row.match_kind)))


def _count_planned_symbol(root: Path, row: PlannedSymbolRowV1) -> int:
    path = root / row.path
    if not path.is_file() or path.suffix != ".py":
        return 0
    tree = _parse_python_source(root, path)
    if row.kind == "class":
        return sum(
            isinstance(node, ast.ClassDef) and node.name == row.symbol
            for node in ast.walk(tree)
        )
    if row.kind in {"function", "method"}:
        return sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == row.symbol
            for node in ast.walk(tree)
        )
    return sum(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == row.symbol
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
        )
        for node in ast.walk(tree)
    )


def _source_symbol_counts(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in _source_paths(root):
        tree = _parse_python_source(root, path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                counts[node.name] += 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                counts[node.name] += 1
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        counts[target.id] += 1
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                counts[node.target.id] += 1
    return counts


def _target_definition_expectations(
    root: Path,
    planned: PlannedSymbolInventoryV1,
) -> dict[str, int]:
    expectations: dict[str, int] = {}
    for row in planned.rows:
        if row.status not in {"migrated", "retained_boundary"}:
            continue
        count = _count_planned_symbol(root, row)
        if count != row.expected_count:
            raise InventoryContractError("planned_symbol_count_drift")
        existing = expectations.get(row.symbol)
        if existing is not None and existing != row.expected_count:
            raise InventoryContractError("target_type_replacement_count_drift")
        expectations[row.symbol] = row.expected_count
    return expectations


def _parse_python_source(root: Path, path: Path) -> ast.Module:
    relative = path.relative_to(root).as_posix()
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise SourceScanError(relative, exc.lineno, exc.msg) from exc
    except UnicodeDecodeError as exc:
        raise SourceScanError(relative, None, exc.reason) from exc


def _source_paths(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for directory in ("src", "scripts", "tests"):
        paths.extend(sorted((root / directory).rglob("*.py")))
    return tuple(paths)


class CliNamespace(argparse.Namespace):
    inventory: bool
    phase: int | None
    require_complete: bool
    root: Path
    fixture: Path | None
    planned_fixture: Path | None

    def __init__(self) -> None:
        super().__init__()
        self.inventory = False
        self.phase = None
        self.require_complete = False
        self.root = ROOT
        self.fixture = None
        self.planned_fixture = None


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    _ = mode.add_argument("--inventory", action="store_true")
    _ = mode.add_argument("--phase", type=int)
    _ = mode.add_argument("--require-complete", action="store_true")
    _ = parser.add_argument("--root", type=Path, default=ROOT)
    _ = parser.add_argument("--fixture", type=Path)
    _ = parser.add_argument("--planned-fixture", type=Path)
    args = parser.parse_args(namespace=CliNamespace())
    fixture = (
        args.fixture or args.root / "tests/fixtures/request_consumer_migration.v1.json"
    )
    planned = (
        args.planned_fixture
        or args.root / "tests/fixtures/planned_symbol_ownership.v1.json"
    )
    try:
        validate_inventory(
            args.root, fixture, planned, args.phase, args.require_complete
        )
    except (InventoryContractError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    fixture_digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    planned_digest = hashlib.sha256(planned.read_bytes()).hexdigest()
    print(
        "OK: {} consumer rows sha256={}; {} planned symbols sha256={}".format(
            len(_load_inventory(fixture).rows),
            fixture_digest,
            len(_load_planned(planned).rows),
            planned_digest,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PlannedSymbolInventoryV1",
    "collect_git_baseline",
    "scan_consumers",
]
