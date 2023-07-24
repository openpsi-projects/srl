from multiprocessing.shared_memory import SharedMemory
from typing import Optional, List, Tuple, Literal
import copy
import dataclasses
import json
import logging
import multiprocessing as mp
import numpy as np
import time

import rlsrl.base.lock as lock
import rlsrl.base.namedarray as namedarray
import rlsrl.base.names as names
import rlsrl.base.numpy_utils as numpy_utils
import rlsrl.base.name_resolve as name_resolve

logger = logging.getLogger("SharedMemory")


class _SharedMemoryDock:

    def __init__(
        self,
        qsize: int,
        shm_names: List[str],
        keys: List[str],
        dtypes: List[np.dtype],
        shapes: List[Tuple[int]],
        second_dim_index:
        bool = True  # whether to use second dimension for batch indices
        # should be true for sample stream, false for inference stream
    ):
        # Initialize shared memory. Create if not exist, else attach to existing shared memory block.
        self._shms = []
        for shm_name, dtype, shape in zip(shm_names, dtypes, shapes):
            if dtype is None:
                self._shms.append(None)
                continue
            try:
                # logger.info(F"Creating shared memory block {qsize}, {dtype}, {shape}")
                shm = SharedMemory(name=shm_name,
                                   create=True,
                                   size=qsize *
                                   numpy_utils.dtype_to_num_bytes(dtype) *
                                   np.prod(shape))
            except FileExistsError:
                shm = SharedMemory(name=shm_name, create=False)
            self._shms.append(shm)

        self._keys = keys
        self._dtypes = dtypes
        self._shapes = shapes
        self._second_dim_index = second_dim_index

        if self._second_dim_index:
            self._buffer = [
                np.frombuffer(shm.buf, dtype=dtype).reshape(
                    shape[0], qsize, *shape[1:]) if dtype is not None else None
                for shm, dtype, shape in zip(self._shms, dtypes, shapes)
            ]
        else:
            self._buffer = [
                np.frombuffer(shm.buf, dtype=dtype).reshape(qsize, *shape)
                if dtype is not None else None
                for shm, dtype, shape in zip(self._shms, dtypes, shapes)
            ]

        logger.debug("_SharedMemoryDock buffer initialized.")

    def put(self, idx, x: namedarray.NamedArray, sort=True):
        if isinstance(idx, np.ndarray) or isinstance(idx, list):
            if sort:
                idx = np.sort(idx)
        flattened_x = namedarray.flatten(x)
        assert len(flattened_x) == len(self._buffer)
        for (k, v), buf in zip(flattened_x, self._buffer):
            if buf is None:
                assert v is None, f"Buffer is none but value is not none, {k}"
            else:
                if self._second_dim_index:
                    # Following the convention of base.buffer, we regard the second dim as batch dimension.
                    buf[:, idx] = v
                else:
                    buf[idx] = v

    def get(self, idx):
        if isinstance(idx, np.ndarray) or isinstance(idx, list):
            if len(idx) == 0:
                raise Exception("Cannot get with empty list idx!")
            idx = np.sort(idx)
        # idx here could be list or int
        if self._second_dim_index:
            xs = [(key, buf[:, idx]) if buf is not None else (key, None)
                  for buf, key in zip(self._buffer, self._keys)]
        else:
            xs = [(key, buf[idx]) if buf is not None else (key, None)
                  for buf, key in zip(self._buffer, self._keys)]
        res = namedarray.from_flattened(xs)
        # logger.info("shared memory get time %f", time.monotonic() - st)
        return res

    def get_key(self, key, idx=None):
        """Return numpy array with key. """
        buffer_idx = self._keys.index(key)
        if idx is None:
            return self._buffer[buffer_idx]
        else:
            return self._buffer[buffer_idx][idx]

    def put_key(self, key, idx, value):
        """Put numpy array with key. """
        buffer_idx = self._keys.index(key)
        self._buffer[buffer_idx][idx] = value

    def close(self):
        # Try release all shared memory blocks.
        self._buffer = []
        for shm in self._shms:
            if not shm:
                continue
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shms = []


class SharedMemoryControl:
    _indicator_names = [
        "_is_writable", "_is_readable", "_is_being_read", "_is_being_written",
        "_time_stamp", "_reuses"
    ]

    def __init__(self,
                 experiment_name,
                 trial_name,
                 ctrl_name,
                 qsize,
                 reuses=1):
        self._qsize = qsize
        self._max_reuse = reuses

        name_prefix = names.shared_memory(experiment_name, trial_name,
                                          ctrl_name)
        name_prefix = name_prefix.strip('/').replace("/", "_")

        self._shms = []
        _created = dict()
        dtypes = [np.uint8] * 4 + [np.int64] * 2
        for indi_name, dtype in zip(SharedMemoryControl._indicator_names,
                                    dtypes):
            try:
                shm = SharedMemory(name=f"{name_prefix}{indi_name}",
                                   create=True,
                                   size=qsize *
                                   numpy_utils.dtype_to_num_bytes(dtype))
                _created[indi_name] = True
            except FileExistsError:
                shm = SharedMemory(name=f"{name_prefix}{indi_name}",
                                   create=False)
                _created[indi_name] = False
            self._shms.append(shm)

        self._is_writable = np.frombuffer(self._shms[0].buf, dtype=dtypes[0])

        self._is_readable = np.frombuffer(self._shms[1].buf, dtype=dtypes[1])
        self._is_being_read = np.frombuffer(self._shms[2].buf, dtype=dtypes[2])
        self._is_being_written = np.frombuffer(self._shms[3].buf,
                                               dtype=dtypes[3])
        self._time_stamp = np.frombuffer(self._shms[4].buf, dtype=dtypes[4])
        self._reuses = np.frombuffer(self._shms[5].buf, dtype=dtypes[5])

        for indi_name in SharedMemoryControl._indicator_names:
            if _created[indi_name]:
                getattr(self,
                        indi_name)[:] = 1 if indi_name == "_is_writable" else 0

    @property
    def qsize(self):
        return self._qsize

    def close(self):
        # Try release all shared memory blocks.
        for indi_name in self._indicator_names:
            if hasattr(self, indi_name):
                delattr(self, indi_name)
        if hasattr(self, "_shms"):
            for shm in self._shms:
                if not shm:
                    continue
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            self._shms = []


class OutOfOrderSharedMemoryControl(SharedMemoryControl):

    def __init__(self,
                 experiment_name,
                 trial_name,
                 ctrl_name,
                 qsize,
                 reuses=1):
        super().__init__(
            experiment_name=experiment_name,
            trial_name=trial_name,
            ctrl_name=ctrl_name,
            qsize=qsize,
            reuses=reuses,
        )
        self._lock = mp.Condition(mp.Lock())

    def acquire_write(
        self,
        batch_size: int = 1,
        allow_overwrite: bool = True,
        preference: Optional[Literal["uniform", "fresh", "old",
                                     "less_reuses_left",
                                     "more_reuses_left"]] = "old",
    ) -> Optional[List[int]]:
        """Acquire writable slots in shared memory.

        Args:
            batch_size (int): The number of slots to be allocated.
            allow_overwrite (bool): Whether to overwrite slots that have been written but
                not read by consumers. Defaults to True.
            preference (Literal): Determine to overwrite which kind of samples.

        Returns:
            Optional[List[int]]: Slot indices or None if not enough available slots.
        """
        with self._lock:
            writable_slots = np.nonzero(self._is_writable)[0]
            readable_slots = np.nonzero(self._is_readable)[0]

            n_slots = len(writable_slots) if not allow_overwrite else len(
                writable_slots) + len(readable_slots)
            if not allow_overwrite and n_slots < batch_size:
                return None

            slot_ids = list(writable_slots[:batch_size])
            # writable -> busy
            self._is_writable[slot_ids] = 0
            if len(slot_ids) < batch_size:

                if preference == "old":
                    # replace the oldest readable slot, in a FIFO pattern
                    slot_ids_ = readable_slots[np.argsort(
                        self._time_stamp[readable_slots])[:batch_size -
                                                          len(slot_ids)]]
                elif preference == "fresh":
                    slot_ids_ = readable_slots[np.argsort(
                        -self._time_stamp[readable_slots])[:batch_size -
                                                           len(slot_ids)]]
                elif preference == "less_reuses_left":
                    slot_ids_ = readable_slots[np.argsort(
                        self._reuses[readable_slots])[:batch_size -
                                                      len(slot_ids)]]
                elif preference == "more_reuses_left":
                    slot_ids_ = readable_slots[np.argsort(
                        -self._reuses[readable_slots])[:batch_size -
                                                       len(slot_ids)]]
                elif preference == "uniform":
                    slot_ids_ = np.random.choice(readable_slots,
                                                 batch_size - len(slot_ids),
                                                 replace=False)
                else:
                    raise NotImplementedError(
                        f"Unknown write preference {preference}.")

                # readable -> busy
                self._is_readable[slot_ids_] = 0
                self._time_stamp[slot_ids_] = 0
                self._reuses[slot_ids_] = 0
                slot_ids += list(slot_ids_)

            self._is_being_written[slot_ids] = 1
            assert (self._is_writable + self._is_readable +
                    self._is_being_read + self._is_being_written == 1).all()
            return slot_ids

    def release_write(self, slot_ids: List[int]):
        """Release slots that is being written.

        Args:
            slot_ids (List[int]): Slot IDs to be released.
        """
        if np.any(np.array(slot_ids) >= self._qsize) or np.any(
                np.array(slot_ids) < 0):
            raise ValueError(f"Slot ID can only in the interval [0, qsize). "
                             f"Input {slot_ids}, qsize {self._qsize}.")
        with self._lock:
            if not np.all(self._is_being_written[slot_ids]):
                raise RuntimeError(
                    "Can't release slots that are not being written!")
            self._is_readable[slot_ids] = 1
            self._is_being_written[slot_ids] = 0
            self._time_stamp[slot_ids] = time.monotonic_ns()
            self._reuses[slot_ids] = self._max_reuse

    def acquire_read(
        self,
        batch_size: int = 1,
        preference: Literal["uniform", "fresh", "old", "less_reuses_left",
                            "more_reuses_left"] = "fresh",
    ) -> Optional[List[int]]:
        """Acquire readable slot IDs in shared memory.

        Args:
            batch_size (int): The number of slots to be allocated.
            preference (Literal): Determine to read which kind of samples.

        Returns:
            Optional[List[int]]: Slot indices or None if not enough available slots.
        """
        with self._lock:
            readable_slots = np.nonzero(self._is_readable)[0]
            if len(readable_slots) < batch_size:
                return None

            if preference == "old":
                slot_ids = readable_slots[np.argsort(
                    self._time_stamp[readable_slots])[:batch_size]]
            elif preference == "fresh":
                slot_ids = readable_slots[np.argsort(
                    -self._time_stamp[readable_slots])[:batch_size]]
            elif preference == 'less_reuses_left':
                slot_ids = readable_slots[np.argsort(
                    self._reuses[readable_slots])[:batch_size]]
            elif preference == "more_reuses_left":
                slot_ids = readable_slots[np.argsort(
                    -self._reuses[readable_slots])[:batch_size]]
            elif preference == "uniform":
                slot_ids = np.random.choice(readable_slots, replace=False)
            else:
                raise NotImplementedError(
                    f"Unknown read preference {preference}.")

            self._is_readable[slot_ids] = 0
            self._is_being_read[slot_ids] = 1
            assert (self._is_writable + self._is_readable +
                    self._is_being_read + self._is_being_written == 1).all()

            return slot_ids

    def release_read(self, slot_ids: List[int]):
        """Release slots that is being read.

        Args:
            slot_ids (List[int]): Slot IDs to be released.
        """
        # t0 = time.monotonic()
        if np.any(np.array(slot_ids) >= self._qsize) or np.any(
                np.array(slot_ids) < 0):
            raise ValueError(f"Slot ID can only in the interval [0, qsize). "
                             f"Input {slot_ids}, qsize {self._qsize}.")
        with self._lock:
            if not np.all(self._is_being_read[slot_ids]):
                raise RuntimeError(
                    "Can't release slots that are not being read!")
            self._reuses[slot_ids] -= 1
            used_up_slots = np.array(slot_ids)[self._reuses[slot_ids] == 0]
            self._is_writable[used_up_slots] = 1
            self._is_being_read[used_up_slots] = 0
            self._time_stamp[used_up_slots] = 0

            remaining_slots = np.array(slot_ids)[self._reuses[slot_ids] > 0]
            self._is_being_read[remaining_slots] = 0
            self._is_readable[remaining_slots] = 1
            assert (self._is_writable + self._is_readable +
                    self._is_being_read + self._is_being_written == 1).all()


class SharedMemoryAgentRegistry:

    def __init__(self):
        self._registeration_lock = mp.Lock()
        self._num_registered = mp.Value('i', 0)

    @property
    def num_registered(self):
        with self._registeration_lock:
            return self._num_registered.value

    def register_agent(self):
        with self._registeration_lock:
            self._num_registered.value += 1
            return self._num_registered.value - 1


class PinnedSharedMemoryControl(SharedMemoryControl):
    """Pinned shared memory control requires lazy instantiation
    because we don't know the qisze in advance.
    """

    def __init__(
        self,
        experiment_name,
        trial_name,
        ctrl_name,
    ):
        self._experiment_name = experiment_name
        self._trial_name = trial_name
        self._ctrl_name = ctrl_name

        self._instantiation_lock = mp.Lock()
        self._instantiated = False

    def instantiate(self, qsize):
        with self._instantiation_lock:
            if not self._instantiated:
                super().__init__(self._experiment_name, self._trial_name,
                                 self._ctrl_name, qsize)
                self._instantiated = True

    @property
    def instantiated(self):
        return self._instantiated


class PinnedRequestSharedMemoryControl(PinnedSharedMemoryControl):

    def __init__(
        self,
        experiment_name,
        trial_name,
        ctrl_name,
    ):
        super().__init__(experiment_name, trial_name, ctrl_name)
        self._lock = lock.RWLock(max_concurrent_read=1,
                                 max_concurrent_write=2147483647,
                                 priority='read')
        self._agent_registry = SharedMemoryAgentRegistry()

    @property
    def agent_registry(self):
        return self._agent_registry

    def acquire(self):
        # Called by policy workers.
        with self._lock.read_locked():
            available_slots = np.nonzero(self._is_readable)[0]
            self._is_readable[available_slots] = 0
            return available_slots

    def release_indices(
        self,
        slot_ids: List[int],
    ):
        # Called by actor workers.
        with self._lock.write_locked():
            assert all(self._is_readable[slot_ids] == 0)
            self._is_readable[slot_ids] = 1


class PinnedResponseSharedMemoryControl(PinnedSharedMemoryControl):

    def __init__(
        self,
        experiment_name,
        trial_name,
        ctrl_name,
    ):
        super().__init__(experiment_name, trial_name, ctrl_name)
        self._lock = lock.RWLock(max_concurrent_read=2147483647,
                                 max_concurrent_write=2147483647,
                                 priority='write')

    def is_ready(self, slot_ids: List[int]):
        # Called by actor workers.
        with self._lock.read_locked():
            return np.all(self._is_readable[slot_ids])

    def acquire_indices(self, slot_ids: List[int]):
        # Called by actor workers.
        with self._lock.read_locked():
            assert np.all(self._is_readable[slot_ids])
            self._is_readable[slot_ids] = 0

    def release_indices(
        self,
        slot_ids: List[int],
    ):
        # Called by policy workers.
        with self._lock.write_locked():
            assert all(self._is_readable[slot_ids] == 0)
            self._is_readable[slot_ids] = 1


@dataclasses.dataclass
class SharedMemoryInferenceStreamCtrl:
    request_ctrl: PinnedRequestSharedMemoryControl
    response_ctrl: PinnedResponseSharedMemoryControl


def writer_make_shared_memory_dock(x,
                                   qsize,
                                   experiment_name,
                                   trial_name,
                                   dock_name,
                                   second_dim_index=True) -> _SharedMemoryDock:
    raw_name = names.shared_memory(experiment_name, trial_name, dock_name)
    shm_name = raw_name.strip("/").replace("/", "_")
    try:
        # If exists, get shared memory info from name resolve
        json_str = name_resolve.get(raw_name)
        shm_names, keys, dtypes, shapes = json.loads(json_str)
        dtypes = [
            numpy_utils.decode_dtype(s) if s is not None else None
            for s in dtypes
        ]
    except name_resolve.NameEntryNotFoundError:
        # If not, parse shared memory info from input namedarray
        keys, values = zip(*namedarray.flatten(x))
        shm_names = [f"{shm_name}_{key}" for key in keys]
        dtypes = [
            value.dtype if value is not None else None for value in values
        ]
        shapes = [
            value.shape if value is not None else None for value in values
        ]
        dtype_strs = [
            numpy_utils.encode_dtype(dt) if dt is not None else None
            for dt in dtypes
        ]
        try:
            name_resolve.add(name=raw_name,
                             value=json.dumps(
                                 [shm_names, keys, dtype_strs, shapes]),
                             delete_on_exit=True,
                             keepalive_ttl=15)
        except name_resolve.NameEntryExistsError:
            pass
    # Initialize shared memory dock
    return _SharedMemoryDock(qsize, shm_names, keys, dtypes, shapes,
                             second_dim_index)


class NothingToRead(Exception):
    pass


def reader_make_shared_memory_dock(qsize,
                                   experiment_name,
                                   trial_name,
                                   dock_name,
                                   second_dim_index=True,
                                   timeout=None) -> _SharedMemoryDock:
    raw_name = names.shared_memory(experiment_name, trial_name, dock_name)
    try:
        dock_info = name_resolve.wait(raw_name, timeout=timeout)
    except TimeoutError:
        return None
    logger.debug("SharedMemoryReader name resolve done.")
    try:
        shm_names, keys, dtypes, shapes = json.loads(dock_info)
        dtypes = [
            numpy_utils.decode_dtype(s) if s is not None else None
            for s in dtypes
        ]
        # Initialize shared memory dock.
        return _SharedMemoryDock(qsize, shm_names, keys, dtypes, shapes,
                                 second_dim_index)
    except Exception as e:
        logger.error(
            "SharedMemoryReader failed to initialize shared memory dock.")
        raise e


class SharedMemoryDock:

    def __init__(self,
                 experiment_name,
                 trial_name,
                 dock_name,
                 qsize,
                 ctrl: Optional[SharedMemoryControl] = None,
                 second_dim_index=True):
        self.__ctrl = ctrl

        self._qsize = qsize
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__dock_name = dock_name
        self.__second_dim_index = second_dim_index

        self._shm_dock = None
        self._role = None

    @property
    def qsize(self):
        return self._qsize

    def _assert_role(self, role):
        if self._role is None:
            self._role = role
        elif self._role != role:
            raise RuntimeError(
                "SharedMemoryDock can't be used as both writer and reader.")

    def write(self, x: namedarray.NamedArray, **kwargs):
        self._assert_role('writer')
        if self._shm_dock is None:  # the first call, initialize shared memory
            self._shm_dock = writer_make_shared_memory_dock(
                x,
                self._qsize,
                self.__experiment_name,
                self.__trial_name,
                self.__dock_name,
                second_dim_index=self.__second_dim_index)

        assert isinstance(self.__ctrl, OutOfOrderSharedMemoryControl)

        while True:
            slot_ids = self.__ctrl.acquire_write(batch_size=1,
                                                 allow_overwrite=False)
            if slot_ids is not None:
                break
            else:
                time.sleep(0.005)

        self._shm_dock.put(slot_ids[0], x, sort=True)

        self.__ctrl.release_write(slot_ids)

    def write_indices(self, x, slot_ids):
        # Thread unsafe.
        self._assert_role('writer')
        if self._shm_dock is None:  # the first call, initialize shared memory
            self._shm_dock = writer_make_shared_memory_dock(
                x[0],
                self._qsize,
                self.__experiment_name,
                self.__trial_name,
                self.__dock_name,
                second_dim_index=self.__second_dim_index)

        self._shm_dock.put(slot_ids, x, sort=False)

    def read(self, batch_size=None):
        self._assert_role('reader')
        if self._shm_dock is None:  # Lazy initialize shared memory dock
            try:
                self._shm_dock = reader_make_shared_memory_dock(
                    self._qsize,
                    self.__experiment_name,
                    self.__trial_name,
                    self.__dock_name,
                    second_dim_index=self.__second_dim_index,
                    timeout=0.1)
            except TimeoutError:
                raise NothingToRead()

        assert isinstance(self.__ctrl, OutOfOrderSharedMemoryControl)

        slot_ids = self.__ctrl.acquire_read(batch_size=(batch_size or 1))
        if slot_ids is None:
            raise NothingToRead()

        if batch_size:
            x = self._shm_dock.get(slot_ids)
        else:
            x = self._shm_dock.get(slot_ids[0])
            x = copy.deepcopy(x)

        self.__ctrl.release_read(slot_ids)
        return x

    def read_indices(self, slot_ids):
        # Thread unsafe.
        self._assert_role('reader')
        if self._shm_dock is None:  # Lazy initialize shared memory dock
            self._shm_dock = reader_make_shared_memory_dock(
                self._qsize,
                self.__experiment_name,
                self.__trial_name,
                self.__dock_name,
                second_dim_index=self.__second_dim_index)
        x = self._shm_dock.get(slot_ids)
        return x

    def close(self):
        if self._shm_dock is not None:
            self._shm_dock.close()
