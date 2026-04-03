from __future__ import annotations


class RetroInferGpuRuntime:
    """
    Placeholder runtime facade for GPU-resident metadata/buffers.

    The current integration delegates the heavy lifting to RetrievalAttention's own
    runtime object, but keeping this facade makes the backend/session/planner split
    match the intended architecture and gives us one place to move future GPU metadata
    ownership into.
    """

    def __init__(self):
        self.active_session_key: tuple[int, ...] | None = None

    def bind_session(self, session_key: tuple[int, ...]) -> None:
        self.active_session_key = session_key

    def clear(self) -> None:
        self.active_session_key = None
