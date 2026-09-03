"""Small torch-free iteration helpers shared by the tokenization and training code."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator


def prefetch_iter[T](iterable: Iterable[T], depth: int = 2) -> Iterator[T]:
    """Iterate ``iterable`` in a background thread, keeping up to ``depth`` items ready.

    Exceptions raised by the producer are re-raised in the consumer; abandoning the
    consumer stops the producer.
    """
    items: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def produce():
        try:
            for item in iterable:
                while not stop.is_set():
                    try:
                        items.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            items.put(done)
        except BaseException as err:  # noqa: BLE001 - forwarded to the consumer
            items.put(err)

    thread = threading.Thread(target=produce, name="prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = items.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
