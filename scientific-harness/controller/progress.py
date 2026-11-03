"""Write flushed, chronological CLI progress without changing machine-readable stdout."""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

logger = logging.getLogger("controller.progress")
logger.addHandler(logging.NullHandler())


@contextmanager
def terminal_progress() -> Iterator[None]:
    """Enable elapsed-time progress on stderr for one CLI invocation, then restore logging."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ProgressFormatter())
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


class _ProgressFormatter(logging.Formatter):
    """Keep messages from controller threads on separate, elapsed-time-stamped lines."""

    def __init__(self) -> None:
        super().__init__()
        self.started = perf_counter()

    def format(self, record: logging.LogRecord) -> str:
        minutes, seconds = divmod(int(perf_counter() - self.started), 60)
        # Proposal rationales and operator paths must not inject terminal controls or log lines.
        message = "".join(
            character if character.isprintable() else " " for character in record.getMessage()
        )
        message = " ".join(message.split())
        return f"[{minutes:02d}:{seconds:02d}] {message}"
