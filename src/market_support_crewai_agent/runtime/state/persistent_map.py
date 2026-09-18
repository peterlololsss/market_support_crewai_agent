from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Generic, TypeVar, override

K = TypeVar("K")
V = TypeVar("V")


class FrozenMapV1(Mapping[K, V], Generic[K, V]):
    __slots__: tuple[str, ...] = ("_view",)

    def __init__(self, items: Mapping[K, V] | tuple[tuple[K, V], ...] = ()) -> None:
        pairs = tuple(items.items()) if isinstance(items, Mapping) else items
        copied: dict[K, V] = {}
        for key, value in pairs:
            if key in copied:
                raise ValueError("FrozenMapV1 rejects duplicate keys")
            copied[key] = value
        self._view: Mapping[K, V] = MappingProxyType(copied)

    @override
    def __getitem__(self, key: K) -> V:
        return self._view[key]

    @override
    def __iter__(self) -> Iterator[K]:
        return iter(sorted(self._view, key=repr))

    @override
    def __len__(self) -> int:
        return len(self._view)

    def with_updates(self, items: Mapping[K, V]) -> FrozenMapV1[K, V]:
        updated = dict(self._view)
        updated.update(items)
        return FrozenMapV1(updated)

    def without_keys(self, keys: frozenset[K]) -> FrozenMapV1[K, V]:
        return FrozenMapV1(
            tuple((key, value) for key, value in self._view.items() if key not in keys)
        )
