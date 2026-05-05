from __future__ import annotations

import importlib
from collections.abc import Callable

from sglang.srt.layers.attention.sparse_framework.selection_spec import (
    CustomSelectionSpec,
)


class SelectionCallbackRegistry:
    _callbacks: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, fn: Callable) -> None:
        cls._callbacks[name] = fn

    @classmethod
    def get(cls, name: str) -> Callable | None:
        return cls._callbacks.get(name)

    @classmethod
    def resolve_parts(
        cls,
        *,
        name: str | None = None,
        import_path: str | None = None,
        fn: str | None = None,
    ) -> Callable:
        if name:
            callback = cls.get(name)
            if callback is not None:
                return callback
        path = import_path or fn
        if path:
            module_name, attr_name = path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        raise ValueError(
            "Custom selection requires either a registered name or an import path."
        )

    @classmethod
    def resolve(cls, spec: CustomSelectionSpec) -> Callable:
        return cls.resolve_parts(
            name=spec.name,
            import_path=spec.import_path,
            fn=spec.fn,
        )


def register_selection_callback(name: str, fn: Callable) -> None:
    SelectionCallbackRegistry.register(name, fn)
