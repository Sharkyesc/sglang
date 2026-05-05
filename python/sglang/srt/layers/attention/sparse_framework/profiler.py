from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseProfilerConfig:
    enabled: bool = False
    output_dir: str = "/tmp/sglang_sparse_framework_profile"
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 1
    with_stack: bool = True
    record_shapes: bool = False
    profile_memory: bool = False
    activities: tuple[str, ...] = ("CPU", "GPU")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SparseProfilerConfig":
        if not data:
            return cls()
        activities = data.get("activities", cls.activities)
        if isinstance(activities, str):
            activities = tuple(item.strip() for item in activities.split(",") if item)
        return cls(
            enabled=bool(data.get("enabled", False)),
            output_dir=str(data.get("output_dir", cls.output_dir)),
            wait=int(data.get("wait", cls.wait)),
            warmup=int(data.get("warmup", cls.warmup)),
            active=int(data.get("active", cls.active)),
            repeat=int(data.get("repeat", cls.repeat)),
            with_stack=bool(data.get("with_stack", cls.with_stack)),
            record_shapes=bool(data.get("record_shapes", cls.record_shapes)),
            profile_memory=bool(data.get("profile_memory", cls.profile_memory)),
            activities=tuple(str(item).upper() for item in activities),
        )


class SparseFrameworkProfiler:
    def __init__(self, config: SparseProfilerConfig):
        self.config = config
        self._profiler = None
        self.step_count = 0
        if not config.enabled:
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if "GPU" in config.activities and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if "CPU" not in config.activities:
            activities = [
                activity
                for activity in activities
                if activity != torch.profiler.ProfilerActivity.CPU
            ]
        if not activities:
            logger.warning("Sparse framework profiler enabled with no activities.")
            return

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=config.wait,
                warmup=config.warmup,
                active=config.active,
                repeat=config.repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.output_dir),
            with_stack=config.with_stack,
            record_shapes=config.record_shapes,
            profile_memory=config.profile_memory,
        )
        self._profiler.start()
        logger.info(
            "Sparse framework profiler started: output_dir=%s activities=%s "
            "wait=%s warmup=%s active=%s repeat=%s",
            config.output_dir,
            tuple(activity.name for activity in activities),
            config.wait,
            config.warmup,
            config.active,
            config.repeat,
        )

    @property
    def enabled(self) -> bool:
        return self._profiler is not None

    def record(self, name: str):
        if not self.enabled:
            return nullcontext()
        return torch.profiler.record_function(name)

    def step(self) -> None:
        if not self.enabled:
            return
        self._profiler.step()
        self.step_count += 1

    def state(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "output_dir": self.config.output_dir,
            "step": self.step_count,
        }

    def stop(self) -> None:
        if self._profiler is None:
            return
        self._profiler.stop()
        self._profiler = None
