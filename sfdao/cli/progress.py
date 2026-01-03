"""Progress reporting helpers for CLI workflows."""

from __future__ import annotations

import atexit
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Iterable, Iterator, Optional, TypeVar

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

T = TypeVar("T")


@dataclass
class AuditProgressConfig:
    total_steps: int
    quiet: bool
    no_progress: bool
    status_interval: float
    verbose: bool


class StatusHeartbeat:
    """Emit periodic status updates while a phase is running."""

    def __init__(self, console: Console, label: str, interval: float, enabled: bool) -> None:
        self._console = console
        self._label = label
        self._interval = interval
        self._enabled = enabled and interval > 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._atexit_registered = False

    def __enter__(self) -> "StatusHeartbeat":
        if not self._enabled:
            return self
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._register_atexit()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            elapsed = 0.0 if self._start_time is None else time.monotonic() - self._start_time
            self._console.print(f"  ... {self._label} ({elapsed:.0f}s elapsed)")

    def _register_atexit(self) -> None:
        if self._atexit_registered or not self._enabled:
            return
        atexit.register(self.stop)
        self._atexit_registered = True

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._interval)


class AuditProgress:
    """Track audit phases and provide optional progress indicators."""

    def __init__(self, console: Console, config: AuditProgressConfig) -> None:
        self._console = console
        self._config = config
        self._current_step = 0

    def start_phase(self, name: str, detail: Optional[str] = None) -> StatusHeartbeat:
        if self._config.quiet:
            return StatusHeartbeat(self._console, name, 0, False)

        label = name if detail is None else f"{name} ({detail})"

        if self._config.no_progress:
            self._console.print(f"Phase: {label}")
        else:
            self._current_step += 1
            self._console.print(f"Step {self._current_step}/{self._config.total_steps}: {label}")

        return StatusHeartbeat(
            self._console,
            label,
            self._config.status_interval,
            enabled=not self._config.no_progress,
        )

    def log_verbose(self, message: str) -> None:
        if self._config.quiet or not self._config.verbose:
            return
        self._console.print(f"  [dim]{message}[/dim]")

    @contextmanager
    def track(self, iterable: Iterable[T], description: str) -> Iterator[Iterable[T]]:
        if self._config.quiet or self._config.no_progress:
            yield iterable
            return

        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

        columns = [
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ]

        with Progress(*columns, console=self._console, transient=True) as progress:
            task_id = progress.add_task(description, total=total)
            yield _ProgressIterable(progress, task_id, iterable)


class _ProgressIterable(Iterable[T]):
    def __init__(self, progress: Progress, task_id: TaskID, iterable: Iterable[T]) -> None:
        self._progress = progress
        self._task_id = task_id
        self._iterable = iterable

    def __iter__(self) -> Iterator[T]:
        for item in self._iterable:
            yield item
            self._progress.advance(self._task_id)
