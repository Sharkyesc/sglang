from __future__ import annotations


class BaseSparseOp:
    def prepare(self, ctx, state: dict) -> None:
        return None

    def run(self, ctx, state: dict):
        raise NotImplementedError
