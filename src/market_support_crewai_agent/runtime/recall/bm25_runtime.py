from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from inspect import signature
from types import ModuleType
from typing import Final

from pydantic import TypeAdapter, ValidationError

type _RawRetrieveRows = tuple[Iterable[Iterable[int]], Iterable[Iterable[float]]]
_RAW_RETRIEVE_ROWS_ADAPTER: Final = TypeAdapter[_RawRetrieveRows](_RawRetrieveRows)


@dataclass(frozen=True, slots=True)
class Bm25QueryV1:
    corpus: Sequence[Sequence[str]]
    query_tokens: Sequence[str]
    top_k: int


@dataclass(frozen=True, slots=True)
class RankedBm25DocumentsV1:
    indices: tuple[int, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _Bm25RuntimeMethods:
    index: Callable[[Sequence[Sequence[str]]], None]
    retrieve: Callable[[Sequence[str], int], RankedBm25DocumentsV1 | None]


def rank_bm25(query: Bm25QueryV1) -> RankedBm25DocumentsV1 | None:
    methods = _load_bm25_methods(len(query.corpus))
    if methods is None:
        return None
    try:
        methods.index(query.corpus)
    except (IndexError, TypeError, ValueError):
        return None
    return methods.retrieve(query.query_tokens, query.top_k)


def normalize_bm25_score(score: float) -> float:
    return min(1.0, score / (score + 1.0))


def _load_bm25_methods(entry_count: int) -> _Bm25RuntimeMethods | None:
    try:
        module = import_module("bm25s")
    except ImportError:
        return None
    factory = _bm25_factory(module, entry_count)
    if factory is None:
        return None
    return factory()


def _bm25_factory(
    module: ModuleType,
    entry_count: int,
) -> Callable[[], _Bm25RuntimeMethods | None] | None:
    candidate = vars(module).get("BM25")
    if not callable(candidate) or not _callable_accepts_no_args(candidate):
        return None

    def build_methods() -> _Bm25RuntimeMethods | None:
        try:
            retriever = candidate()
        except (TypeError, ValueError):
            return None
        if not isinstance(retriever, Hashable):
            return None
        return _bm25_methods_from(retriever, entry_count)

    return build_methods


def _bm25_methods_from(
    retriever: Hashable,
    entry_count: int,
) -> _Bm25RuntimeMethods | None:
    members = vars(type(retriever))
    index_method = members.get("index")
    retrieve_method = members.get("retrieve")
    if (
        not callable(index_method)
        or not callable(retrieve_method)
        or not _callable_accepts_index_args(index_method, retriever)
        or not _callable_accepts_retrieve_args(retrieve_method, retriever)
    ):
        return None
    return _Bm25RuntimeMethods(
        index=_index_runner(index_method, retriever),
        retrieve=_retrieve_runner(retrieve_method, retriever, entry_count),
    )


def _index_runner[CallableReturn](
    index_method: Callable[..., CallableReturn], retriever: Hashable
) -> Callable[[Sequence[Sequence[str]]], None]:
    def run_index(corpus: Sequence[Sequence[str]]) -> None:
        _ = index_method(retriever, corpus, show_progress=False)

    return run_index


def _retrieve_runner[CallableReturn](
    retrieve_method: Callable[..., CallableReturn],
    retriever: Hashable,
    entry_count: int,
) -> Callable[[Sequence[str], int], RankedBm25DocumentsV1 | None]:
    def run_retrieve(
        query_tokens: Sequence[str], top_k: int
    ) -> RankedBm25DocumentsV1 | None:
        try:
            result = retrieve_method(
                retriever, [query_tokens], k=top_k, show_progress=False
            )
            document_rows, score_rows = _RAW_RETRIEVE_ROWS_ADAPTER.validate_python(
                result
            )
        except (IndexError, TypeError, ValueError, ValidationError):
            return None
        return _parse_ranked_documents(
            document_rows,
            score_rows,
            entry_count=entry_count,
        )

    return run_retrieve


def _callable_accepts_no_args[ReturnT](factory: Callable[..., ReturnT]) -> bool:
    try:
        _ = signature(factory).bind()
    except (TypeError, ValueError):
        return False
    return True


def _callable_accepts_index_args[CallableReturn](
    index_method: Callable[..., CallableReturn], retriever: Hashable
) -> bool:
    try:
        _ = signature(index_method).bind(retriever, (("token",),), show_progress=False)
    except (TypeError, ValueError):
        return False
    return True


def _callable_accepts_retrieve_args[CallableReturn](
    retrieve_method: Callable[..., CallableReturn], retriever: Hashable
) -> bool:
    try:
        _ = signature(retrieve_method).bind(
            retriever,
            (("token",),),
            k=1,
            show_progress=False,
        )
    except (TypeError, ValueError):
        return False
    return True


def _parse_ranked_documents(
    document_rows: Iterable[Iterable[int]],
    score_rows: Iterable[Iterable[float]],
    *,
    entry_count: int,
) -> RankedBm25DocumentsV1 | None:
    try:
        raw_document_row = next(iter(document_rows))
        raw_score_row = next(iter(score_rows))
        indices = tuple(int(raw_index) for raw_index in raw_document_row)
        scores = tuple(float(raw_score) for raw_score in raw_score_row)
    except (StopIteration, TypeError, ValueError, ValidationError):
        return None
    if len(indices) != len(scores):
        return None
    if any(index < 0 or index >= entry_count for index in indices):
        return None
    return RankedBm25DocumentsV1(indices=indices, scores=scores)
