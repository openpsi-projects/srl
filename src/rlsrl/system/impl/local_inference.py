from typing import Optional, Union
import binascii
import dataclasses
import logging
import numpy as np
import pickle
import time
import threading

import rlsrl.api.config as config_api
import rlsrl.api.policy as policy_api
import rlsrl.base.name_resolve as name_resolve
import rlsrl.base.namedarray as namedarray
import rlsrl.base.names as names
import rlsrl.base.shared_memory as shared_memory
import rlsrl.system.api.inference_stream as inference_stream

logger = logging.getLogger("LocalInferenceStream")


class PinnedSharedMemoryInferenceClient(inference_stream.InferenceClient):
    """ Inference client that uses shared memory to communicate with the server.
    """

    def __init__(
        self,
        experiment_name,
        trial_name,
        stream_name,
        ctrl: shared_memory.SharedMemoryInferenceStreamCtrl,
    ):
        """Init method of ip inference client.

        Args:
            address of on the server.
        """
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__stream_name = stream_name

        self.__request_ctrl = ctrl.request_ctrl
        self.__response_ctrl = ctrl.response_ctrl
        # Docks will be lazily initialized after agent registeration.
        self.__request_dock: Optional[shared_memory.SharedMemoryDock] = None
        self.__response_dock: Optional[shared_memory.SharedMemoryDock] = None

        # Client_id is a randomly generated non-zero np.int32
        self.client_id = np.random.randint(1, 2147483647)
        logger.info(
            f"Generated client id {self.client_id} for shared memory inference client."
        )
        self._request_count = 0
        self.__request_buffer = []
        self.__index_buffer = []

        self.__req_id_generation_lock = threading.Lock(
        )  # Lock self._request_count for safe id generation.
        # Lock self.__request_buffer for concurrent writing/flushing.
        self.__request_buffer_lock = threading.Lock()

    def __post_request_with_id(self, request, req_id, index):
        request.request_id = np.array([req_id], dtype=np.int32)
        request.client_id = np.array([self.client_id], dtype=np.int32)
        request.buffer_index = np.array([index], dtype=np.int32)
        with self.__request_buffer_lock:
            self.__request_buffer.append(request)
            self.__index_buffer.append(index)

    def post_request(self, request: policy_api.RolloutRequest,
                     index: int) -> int:
        """Buffer a request and get a new request id (Thread Safe).
        Index is used to identify which agent is posting the request.
        """
        with self.__req_id_generation_lock:
            req_id = self._request_count
            self._request_count += 1

        logger.debug(f"Generated req_id {req_id}")
        self.__post_request_with_id(request, req_id, index)
        return req_id

    def is_ready(self, inf_id, inf_indices) -> bool:
        return self.__response_ctrl.is_ready(inf_indices)

    def consume_result(self, inference_ids, inf_indices):
        self.__response_ctrl.acquire_indices(inf_indices)
        resp = self.__response_dock.read_indices(inf_indices)
        return [resp[i] for i in range(len(inf_indices))]

    def register_agent(self):
        # Called in worker configuration
        return self.__request_ctrl.agent_registry.register_agent()

    def _create_dock(self):
        qsize = self.__request_ctrl.agent_registry.num_registered
        self.__request_ctrl.instantiate(qsize=qsize)
        self.__request_dock = shared_memory.SharedMemoryDock(
            self.__experiment_name,
            self.__trial_name,
            self.__stream_name + "_inf_request",
            qsize,
            second_dim_index=False)
        self.__response_ctrl.instantiate(qsize=qsize)
        self.__response_dock = shared_memory.SharedMemoryDock(
            self.__experiment_name,
            self.__trial_name,
            self.__stream_name + "_inf_response",
            qsize,
            second_dim_index=False)
        try:
            name_resolve.add(
                names.pinned_shm_qsize(self.__experiment_name,
                                       self.__trial_name, self.__stream_name),
                str(qsize),
                keepalive_ttl=15,
                delete_on_exit=True,
                replace=False,
            )
        except name_resolve.NameEntryExistsError:
            pass

    def flush(self):
        # Called in main thread.
        if len(self.__request_buffer) == 0:
            return
        with self.__request_buffer_lock:
            indices = self.__index_buffer
            agg_request = namedarray.recursive_aggregate(
                self.__request_buffer, np.stack)
            self.__request_buffer = []
            self.__index_buffer = []

        if self.__request_dock is None:
            self._create_dock()

        assert (np.array(indices) == agg_request.buffer_index[:, 0]).all(
        ), f"Put indices do not match {indices}, {agg_request.buffer_index[:, 0]}."
        self.__request_dock.write_indices(agg_request, indices)
        self.__request_ctrl.release_indices(indices)

    def poll_responses(self):
        pass

    def get_constant(self, name, timeout=5):
        constant_name = names.inference_stream_constant(
            experiment_name=self.__experiment_name,
            trial_name=self.__trial_name,
            stream_name=self.__stream_name,
            constant_name=name)
        st = time.time()
        while time.time() - st < timeout:
            try:
                r = name_resolve.get(constant_name).encode()
                break
            except name_resolve.NameEntryNotFoundError:
                time.sleep(0.1)
        return pickle.loads(binascii.a2b_base64(r))

    def __del__(self):
        if self.__request_dock is not None:
            self.__request_dock.close()
        if self.__response_dock is not None:
            self.__response_dock.close()


inference_stream.register_client(
    config_api.InferenceStream.Type.SHARED_MEMORY,
    lambda spec, ctrl: PinnedSharedMemoryInferenceClient(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
        ctrl=ctrl,
    ),
)


class PinnedSharedMemoryInferenceServer(inference_stream.InferenceServer):
    # dock server, reader, writer
    def __init__(
        self,
        experiment_name,
        trial_name,
        stream_name,
        ctrl: shared_memory.SharedMemoryInferenceStreamCtrl,
    ):
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__stream_name = stream_name

        self.__request_ctrl = ctrl.request_ctrl
        self.__response_ctrl = ctrl.response_ctrl
        # Docks will be lazily initialized after agent registeration.
        self.__request_dock: Optional[shared_memory.SharedMemoryDock] = None
        self.__response_dock: Optional[shared_memory.SharedMemoryDock] = None

    def _create_dock(self):
        # Initialize request shared memory dock
        qsize = self.__request_ctrl.agent_registry.num_registered
        self.__request_ctrl.instantiate(qsize=qsize)
        self.__request_dock = shared_memory.SharedMemoryDock(
            self.__experiment_name,
            self.__trial_name,
            self.__stream_name + "_inf_request",
            qsize,
            second_dim_index=False)
        self.__response_ctrl.instantiate(qsize=qsize)
        self.__response_dock = shared_memory.SharedMemoryDock(
            self.__experiment_name,
            self.__trial_name,
            self.__stream_name + "_inf_response",
            qsize,
            second_dim_index=False)

    def poll_requests(self):
        """ Called by policy worker. Read all ready requests in request shared memory dock.
        Set received time.
        """
        if self.__request_dock is None:
            try:
                name_resolve.wait(
                    names.pinned_shm_qsize(self.__experiment_name,
                                           self.__trial_name,
                                           self.__stream_name),
                    timeout=1,
                )
            except TimeoutError:
                return []
            self._create_dock()

        available_slots = self.__request_ctrl.acquire()
        if len(available_slots) == 0:
            return []

        batch = self.__request_dock.read_indices(available_slots)
        batch.received_time[:] = time.monotonic_ns()
        return [batch]

    def respond(self, responses: policy_api.RolloutResult):
        """ Respond with an aggregated rollout result, write them into response shared memory dock.
        """
        indices = responses.buffer_index[:, 0]
        self.__response_dock.write_indices(responses, indices)
        self.__response_ctrl.release_indices(indices)

    def set_constant(self, name, value):
        name_resolve.add(
            names.inference_stream_constant(
                experiment_name=self.__experiment_name,
                trial_name=self.__trial_name,
                stream_name=self.__stream_name,
                constant_name=name),
            binascii.b2a_base64(pickle.dumps(value)).decode(),
            keepalive_ttl=30,
            replace=True,
        )

    def __del__(self):
        if self.__request_dock is not None:
            self.__request_dock.close()
        if self.__response_dock is not None:
            self.__response_dock.close()


inference_stream.register_server(
    config_api.InferenceStream.Type.SHARED_MEMORY,
    lambda spec, ctrl: PinnedSharedMemoryInferenceServer(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
        ctrl=ctrl,
    ),
)
