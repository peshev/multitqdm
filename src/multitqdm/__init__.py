"""Thread-safe multi-progress rendering for tqdm.

The worker threads communicate progress through a queue. A dedicated renderer
thread is the only thread that creates or updates tqdm instances, which avoids
concurrent terminal writes from worker threads.
"""

from __future__ import annotations

import itertools
import threading
from concurrent.futures import Executor
from queue import Empty, Queue
from typing import Iterator

from tqdm import tqdm


class BaseEvent:
    """Base class for renderer events."""


class BaseTaskEvent(BaseEvent):
    """Base class for events associated with one progress task."""

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id


class TaskStartedEvent(BaseTaskEvent):
    """Event sent when a task progress bar should be created."""

    def __init__(self, task_id: int, *args, **kwargs) -> None:
        super().__init__(task_id)
        self.args = args
        self.kwargs = kwargs


class TaskProgressedEvent(BaseTaskEvent):
    """Event sent when a task has advanced."""

    def __init__(self, task_id: int, progress: int = 1) -> None:
        super().__init__(task_id)
        self.progress = progress


class TaskCompletedEvent(BaseTaskEvent):
    """Event sent when a task is complete."""


class StoppedEvent(BaseEvent):
    """Event sent when the renderer should shut down."""


def _set_pos(bar: tqdm, pos: int) -> None:
    if bar.pos != -pos:
        bar.clear()
        bar.pos = -pos
        bar.refresh()


class ProgressBarRenderer(threading.Thread):
    """Owns all tqdm instances and renders progress updates."""

    def __init__(self, events: Queue[BaseEvent], *args, total_completed: bool = False, **kwargs):
        super().__init__(name="progress-renderer")
        self._events = events
        self._bars: dict[int, tqdm] = {}
        self._total_bar = tqdm(*args, **{**kwargs, "position": 0, "leave": True})
        self._total_completed = total_completed

    def _update_positions(self):
        for position, (_task_id, bar) in enumerate(sorted(self._bars.items())):
            _set_pos(bar, position)
        _set_pos(self._total_bar, len(self._bars))

    def run(self) -> None:
        try:
            while True:
                try:
                    event = self._events.get(timeout=0.1)
                except Empty:
                    continue

                try:
                    if isinstance(event, StoppedEvent):
                        return
                    if isinstance(event, BaseTaskEvent):
                        self._handle_event(event)
                    else:
                        raise TypeError(f"Unexpected event type {type(event).__name__}")
                finally:
                    self._events.task_done()
        finally:
            for bar in self._bars.values():
                bar.close()
            self._total_bar.close()

    def _handle_event(self, event: BaseTaskEvent) -> None:
        if event.task_id is None:
            raise ValueError("All task events require task_id")

        if isinstance(event, TaskStartedEvent):
            self._bars[event.task_id] = tqdm(
                *event.args,
                **{**event.kwargs, "position": len(self._bars), "leave": True}
            )
            self._update_positions()
            return

        if isinstance(event, TaskProgressedEvent):
            bar = self._bars[event.task_id]
            bar.update(event.progress)
            if not self._total_completed:
                self._total_bar.update(event.progress)
        elif isinstance(event, TaskCompletedEvent):
            bar = self._bars.get(event.task_id)
            if bar:  # If task_id doesn't have a bar, but receives a TaskCompletedEvent, that means that it was skipped
                if bar.total is not None and bar.n < bar.total:
                    remaining = int(bar.total - bar.n)
                    bar.update(remaining)
                    if not self._total_completed:
                        self._total_bar.update(remaining)
                _set_pos(bar, 0)
                bar.close()
                del self._bars[event.task_id]
            if self._total_completed:
                self._total_bar.update(1)
            self._update_positions()
        else:
            raise TypeError(f"Unexpected task event type {type(event).__name__}")


class ProgressBarExecutor:
    """Wrap an Executor and inject a ProgressBarTask into submitted work."""

    def __init__(self, executor: Executor, *args, total_completed=False, **kwargs):
        self.executor = executor
        self.events: Queue[BaseEvent] = Queue()
        self.renderer = ProgressBarRenderer(self.events, *args, total_completed=total_completed, **kwargs)
        self.task_counter = itertools.count()

    def _progressbar_generator(self) -> Iterator[TaskProgressBar]:
        while True:
            yield TaskProgressBar(next(self.task_counter), self.events)

    def submit(self, fn, /, *args, **kwargs):
        if not self.renderer.is_alive():
            self.renderer.start()
        return self.executor.submit(
            fn,
            next(self._progressbar_generator()),
            *args,
            **kwargs
        )

    def map(self, fn, *iterables, timeout=None, chunksize=1, buffersize=None):
        if not self.renderer.is_alive():
            self.renderer.start()
        return self.executor.map(
            fn,
            self._progressbar_generator(),
            *iterables,
            timeout=timeout,
            chunksize=chunksize,
            buffersize=buffersize
        )

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.events.join()
        self.events.put(StoppedEvent())
        self.renderer.join()
        self.executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> "ProgressBarExecutor":
        result = self.executor.__enter__()
        if result is not self.executor:
            # If executor.__enter__() returns something other than self, then the assumption made below is incorrect
            raise RuntimeError("Expected executor to return self as context value, but got something different")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return self.executor.__exit__(exc_type, exc_val, exc_tb)


class ProgressBar:
    def start(self, *args, **kwargs):
        raise NotImplementedError

    def progress(self, progress: int = 1):
        raise NotImplementedError

    def complete(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.complete()


class TaskProgressBar(ProgressBar):
    """Task handle passed to worker functions for progress reporting."""

    def __init__(self, task_id: int, events: Queue[BaseEvent]):
        self.task_id = task_id
        self.events = events

    def start(self, *args, **kwargs):
        self.events.put(TaskStartedEvent(task_id=self.task_id, *args, **kwargs))
        return self

    def progress(self, progress: int = 1):
        self.events.put(TaskProgressedEvent(task_id=self.task_id, progress=progress))

    def complete(self):
        self.events.put(TaskCompletedEvent(task_id=self.task_id))


class SimpleProgressBar(ProgressBar):
    def __init__(self):
        self._pb = None

    def start(self, *args, **kwargs):
        assert self._pb is None
        self._pb = tqdm(*args, **kwargs)

    def progress(self, progress: int = 1):
        assert self._pb is not None
        self._pb.update(progress)

    def complete(self):
        assert self._pb is not None
        self._pb.close()
        self._pb = None
