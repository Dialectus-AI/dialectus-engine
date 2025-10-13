"""
Type stubs for the standard library module `string.templatelib`.

These annotations are used to satisfy Pyright strict mode when running
against Python 3.14, where the new t-string implementation currently lacks
Typeshed coverage.

Upstream patch coming: https://github.com/microsoft/pyright/issues/11003

"""

from __future__ import annotations

from typing import Any, Iterable, Sequence, Tuple


class Interpolation(str):
    """Represents a single interpolation within a template string."""


class Template(str):
    """Runtime type for template string literals (t-strings)."""

    interpolations: Sequence[Interpolation]

    def __new__(cls, *parts: str | Interpolation) -> Template: ...

    def __reduce__(self) -> Tuple[Any, ...]: ...


def convert(obj: Any, conversion: str | None) -> Any: ...


def _template_unpickle(
    strings: Iterable[str],
    interpolations: Iterable[Interpolation],
) -> Template: ...
