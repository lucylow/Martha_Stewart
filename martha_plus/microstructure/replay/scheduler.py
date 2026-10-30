"""Stable virtual-time scheduler for exchange and network events."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(order=True, frozen=True, slots=True)
class ScheduledTask:
    due_ns: int
    sequence: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False, default=None)
    source_ns: int = field(compare=False, default=0)


class EventScheduler:
    """Priority queue ordered by due timestamp, then insertion sequence."""

    def __init__(self, *, start_ns: int = 0) -> None:
        if start_ns < 0:
            raise ValueError("start_ns must be non-negative")
        self.now_ns = start_ns
        self._sequence = 0
        self._heap: list[ScheduledTask] = []
        self.processed = 0

    def schedule(self, due_ns: int, kind: str, payload: Any = None, *, source_ns: int | None = None) -> ScheduledTask:
        if due_ns < self.now_ns:
            raise ValueError("cannot schedule a task in the virtual past")
        if not kind:
            raise ValueError("task kind must not be blank")
        self._sequence += 1
        task = ScheduledTask(due_ns, self._sequence, kind, payload, due_ns if source_ns is None else source_ns)
        heapq.heappush(self._heap, task)
        return task

    def pop(self) -> ScheduledTask:
        if not self._heap:
            raise IndexError("scheduler is empty")
        task = heapq.heappop(self._heap)
        if task.due_ns < self.now_ns:
            raise AssertionError("scheduler produced a task in the past")
        self.now_ns = task.due_ns
        self.processed += 1
        return task

    def peek(self) -> ScheduledTask | None:
        return self._heap[0] if self._heap else None

    def run(self, handler, *, max_tasks: int | None = None, until_ns: int | None = None) -> int:
        if max_tasks is not None and max_tasks < 0:
            raise ValueError("max_tasks must be non-negative")
        count = 0
        while self._heap:
            if max_tasks is not None and count >= max_tasks:
                break
            if until_ns is not None and self._heap[0].due_ns > until_ns:
                break
            task = self.pop()
            handler(task)
            count += 1
        if until_ns is not None and until_ns >= self.now_ns:
            self.now_ns = until_ns
        return count

    def clear(self) -> None:
        self._heap.clear()

    def __len__(self) -> int:
        return len(self._heap)

    def pending(self) -> tuple[ScheduledTask, ...]:
        return tuple(sorted(self._heap))
