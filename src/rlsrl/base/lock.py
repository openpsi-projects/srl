from typing import Literal, Optional
import contextlib
import logging
import multiprocessing as mp

logger = logging.getLogger("Lock")


class RWLock:
    """ A lock object that allows many simultaneous write and many
    simultaneous read, but cannot be acquired for both at the same time. """

    def __init__(self,
                 max_concurrent_read: int = 1,
                 max_concurrent_write: int = 1,
                 priority: Optional[Literal['read', 'write']] = None):
        self._lock = mp.Condition(mp.Lock())

        self._max_read = max_concurrent_read
        self._max_write = max_concurrent_write

        self._n_readers = mp.Value("i", 0)
        self._n_writers = mp.Value("i", 0)

        self._n_waiting_readers = mp.Value("i", 0)
        self._n_waiting_writers = mp.Value("i", 0)

        self._priority = priority
        if priority not in [None, 'read', 'write']:
            raise ValueError("Priority must be one of 'read', 'write'.")

    @property
    def n_waitings(self):
        return self._n_waiting_writers.value + self._n_waiting_readers.value

    def acquire_read(self):
        self._lock.acquire()
        while (self._n_writers.value > 0
               or (self._priority == 'write'
                   and self._n_waiting_writers.value > 0)
               or self._n_readers.value >= self._max_read):
            self._n_waiting_readers.value += 1
            logger.debug(
                f"Waiting for read lock. Numer of waiting readers: {self._n_waiting_readers.value}."
            )
            self._lock.wait()
            self._n_waiting_readers.value -= 1
        try:
            logger.debug(
                f"Acquire read. Concurrent readers: {self._n_readers.value}. "
                f"Numer of waiting readers: {self._n_waiting_readers.value}.")
            self._n_readers.value += 1
            assert self._n_readers.value <= self._max_read
        finally:
            self._lock.release()

    def release_read(self):
        self._lock.acquire()
        try:
            self._n_readers.value -= 1
            logger.debug(
                f"Relese read. Remaining readers: {self._n_readers.value}. "
                f"Numer of waiting readers: {self._n_waiting_readers.value}."
                f"Numer of waiting writers: {self._n_waiting_writers.value}.")
            if self.n_waitings > 0:
                self._lock.notify(self.n_waitings)
        finally:
            self._lock.release()

    def acquire_write(self):
        self._lock.acquire()
        while (self._n_readers.value > 0 or
               (self._priority == 'read' and self._n_waiting_readers.value > 0)
               or self._n_writers.value >= self._max_write):
            self._n_waiting_writers.value += 1
            logger.debug(
                f"Waiting for write lock. "
                f"Current writers {self._n_writers.value}. "
                f"Numer of waiting writers: {self._n_waiting_writers.value}.")
            self._lock.wait()
            self._n_waiting_writers.value -= 1
        try:
            self._n_writers.value += 1
            logger.debug(
                f"Acquire write. Concurrent writers: {self._n_writers.value}. "
                f"Numer of waiting writers: {self._n_waiting_writers.value}.")
            assert self._n_writers.value <= self._max_write
        finally:
            self._lock.release()

    def release_write(self):
        self._lock.acquire()
        try:
            self._n_writers.value -= 1
            logger.debug(
                f"Relese write. Remaining writers: {self._n_writers.value}. "
                f"Numer of waiting readers: {self._n_waiting_readers.value}."
                f"Numer of waiting writers: {self._n_waiting_writers.value}.")
            if self.n_waitings > 0:
                self._lock.notify(self.n_waitings)
        finally:
            self._lock.release()

    @contextlib.contextmanager
    def read_locked(self):
        try:
            self.acquire_read()
            yield
        finally:
            self.release_read()

    @contextlib.contextmanager
    def write_locked(self):
        try:
            self.acquire_write()
            yield
        finally:
            self.release_write()
