from typing import Tuple, List, Union
import binascii
import json
import logging
import numpy as np
import pickle
import time
import threading
import zmq

import rlsrl.api.config as config_api
import rlsrl.api.policy as policy_api
import rlsrl.base.namedarray as namedarray
import rlsrl.base.timeutil as timeutil
import rlsrl.base.name_resolve as name_resolve
import rlsrl.base.network as network
import rlsrl.base.names as names
import rlsrl.system.api.inference_stream as inference_stream

_ROLLOUT_REQUEST_RETRY_INTERVAL_SECONDS = 100.  # 100.

logger = logging.getLogger("RemoteInferenceStream")


class IpInferenceClient(inference_stream.InferenceClient):
    """Inference Client based on IP, Currently implemented in ZMQ.
    By calling client.post_request(msg, flush=False), the inference client will
    cache the the posted request. When client.flush() is called, the inference client will batch
    all the inference request and send to inference server.

    NOTE: Calling client.post_request(msg, flush=True) is discouraged. Sending scattered request may
    overwhelm the inference side.
    """

    def __init__(self, server_addresses: Union[List[str], str],
                 serialization_method: str):
        """Init method of ip inference client.

        Args:
            address of on the server.
        """
        self.__addresses = server_addresses if isinstance(
            server_addresses, List) else [server_addresses]
        logger.debug(f"Client: connecting to servers {self.__addresses}")

        # Client_id is a randomly generated np.int32.
        self.client_id = np.random.randint(0, 2147483647)
        self.__context, self.__socket = self._make_sockets()
        self._request_count = 0
        self.__request_buffer = []

        self._response_cache = {}
        self._request_send_time = {}
        self._pending_requests = {}
        self._retried_requests = set()

        self.__req_id_generation_lock = threading.Lock(
        )  # Lock self._request_count for safe id generation.
        # Lock self.__request_buffer for concurrent writing/ flushing.
        self.__request_buffer_lock = threading.Lock()
        # Locks self._pending_request and self._request_send_time
        self.__request_metadata_lock = threading.Lock()

        self.__retry_frequency_control = timeutil.FrequencyControl(
            frequency_seconds=5)
        self.__debug_frequency_control = timeutil.FrequencyControl(
            frequency_seconds=5)

        self.__serialization_method = serialization_method

        # debug
        self.first_time_req = True
        self.first_time_response = True

    def __del__(self):
        self.__socket.close(linger=False)
        self.__context.destroy(linger=False)

    def _make_sockets(self) -> Tuple[zmq.Context, zmq.Socket]:
        """Setup ZMQ socket for posting observations and receiving actions.
        Sockets are shared across multiple environments in the actor.

        Outbound message:
            [request]

        Inbound message:
            [b"client_id", msg]
        """
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.identity = self.client_id.to_bytes(length=4, byteorder="little")
        for addr in self.__addresses:
            socket.connect(f"tcp://{addr}")
        socket.setsockopt(zmq.LINGER, 0)
        return ctx, socket

    def __post_request_with_id(self, request, req_id):
        request.request_id = np.array([req_id], dtype=np.int64)
        request.client_id = np.array([self.client_id], dtype=np.int32)
        with self.__request_buffer_lock:
            self.__request_buffer.append(request)
        self._request_send_time[req_id] = time.monotonic_ns()
        self._pending_requests[req_id] = request

    def post_request(self, request: policy_api.RolloutRequest, _=None) -> int:
        with self.__req_id_generation_lock:
            req_id = self._request_count
            self._request_count += 1

        logger.debug(f"Generated req_id {req_id}")
        self.__post_request_with_id(request, req_id)
        return req_id

    def is_ready(self, inference_ids, _=None) -> bool:
        check_retry = self.__retry_frequency_control.check(
        )  # If true, we retry expired requests.
        for req_id in inference_ids:
            if req_id not in list(self._response_cache.keys()):
                if check_retry:
                    with self.__request_metadata_lock:
                        expired = req_id in self._request_send_time.keys() and \
                                  (time.monotonic_ns() - self._request_send_time[req_id]) / 1e9 > _ROLLOUT_REQUEST_RETRY_INTERVAL_SECONDS
                        r = self._pending_requests[req_id]
                    if expired:
                        self.__post_request_with_id(r, req_id)
                        logger.info(
                            f"Request with req_id {req_id} timed out. Retrying..."
                        )
                return False
        return True

    def consume_result(self, inference_ids, _=None):
        if self.__debug_frequency_control.check():
            logger.debug("Cached reqs", list(self._response_cache.keys()))
            logger.debug("Pending reqs", list(self._pending_requests.keys()))
        return [self._response_cache.pop(req_id) for req_id in inference_ids]

    def flush(self):
        if len(self.__request_buffer) > 0:
            with self.__request_buffer_lock:
                agg_request = namedarray.recursive_aggregate(
                    self.__request_buffer, np.stack)
                self.__request_buffer = []
            self.__socket.send_multipart(
                namedarray.dumps(agg_request,
                                 method=self.__serialization_method))

    def poll_responses(self):
        """Get all action messages from inference servers. Thread unsafe"""
        try:
            _msg = self.__socket.recv_multipart(zmq.NOBLOCK)
            responses = namedarray.loads(_msg)
            for i in range(responses.length(dim=0)):
                req_id = responses.request_id[i, 0]
                # with self.__consume_lock:
                if req_id in list(self._response_cache.keys()):
                    raise ValueError(
                        "receiving multiple result with request id {}."
                        "Have you specified different inferencer client_ids for actors?"
                        .format(req_id))
                else:
                    if req_id not in list(self._request_send_time.keys()):
                        if req_id in self._retried_requests:
                            logger.warning(
                                f"Received multiple responses for request {req_id}. "
                                f"Request {req_id} has been retried, ignoring this case."
                            )
                            continue
                        # This is impossible.
                        raise RuntimeError(
                            f"Impossible case: got response but I didn't send it? {req_id}"
                        )
                    with self.__request_metadata_lock:
                        latency = (time.monotonic_ns() -
                                   self._request_send_time.pop(req_id)) / 1e9
                        self._pending_requests.pop(req_id)
                    self._response_cache[req_id] = responses[i]
                    logger.debug(
                        f"Response cache: {list(self._response_cache.keys())}")
            return responses.length(dim=0)
        except zmq.ZMQError:
            return 0
        except Exception as e:
            raise e

    def get_constant(self, name):
        raise NotImplementedError()


inference_stream.register_client(
    config_api.InferenceStream.Type.IP,
    lambda spec: IpInferenceClient(
        server_addresses=spec.address,
        serialization_method=spec.serialization_method,
    ),
)


class IpInferenceServer(inference_stream.InferenceServer):
    """InferenceServer inited by IP, Currently implemented in ZMQ pub-sub pattern.
    Be careful only to initialize server within process, as ZMQ context cannot
    be inherited from one process to another
    """

    def __init__(self, address, serialization_method):
        """Init method of Ip inference server.
        Args:
            address: Address of Server
        """
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.ROUTER)
        self.__socket.setsockopt(zmq.LINGER, 0)

        if address == '':
            host_ip = network.gethostip()
            port = self.__socket.bind_to_random_port(f"tcp://{host_ip}")
            address = f"{host_ip}:{port}"
        else:
            self.__socket.bind(f"tcp://{address}")

        logger.debug(f"Server: address {address}")

        self._address = address
        self.__serialization_method = serialization_method

    def __del__(self):
        self.__socket.close()
        self.__context.destroy(linger=False)

    def poll_requests(self, max_iter=64):
        request_batches = []
        for _ in range(max_iter):
            try:
                client_id_, *msg = self.__socket.recv_multipart(zmq.NOBLOCK)
                # Client id is recorded in msg itself, we don't actually need client_id_.
                try:
                    requests = namedarray.loads(msg)
                except pickle.UnpicklingError:
                    logger.info(f"Unpickling request failed. content: {msg}")
                    continue
                requests.received_time[:] = time.monotonic_ns()
                request_batches.append(requests)
            except zmq.ZMQError:
                break
        return request_batches

    def respond(self, responses: policy_api.RolloutResult):
        logger.debug(f"respond to req_ids: {responses.request_id}")
        idx = np.concatenate([[0],
                              np.where(np.diff(responses.client_id[:, 0]))[0] +
                              1, [responses.length(dim=0)]])
        for i in range(len(idx) - 1):
            self.__socket.send_multipart([
                int(responses.client_id[idx[i], 0]).to_bytes(
                    length=4, byteorder="little"),
            ] + namedarray.dumps(responses[idx[i]:idx[i + 1]],
                                 method=self.__serialization_method))

    def set_constant(self, name, value):
        raise NotImplementedError()


inference_stream.register_server(
    config_api.InferenceStream.Type.IP,
    lambda spec: IpInferenceServer(
        address=spec.address,
        serialization_method=spec.serialization_method,
    ),
)


class NameResolvingInferenceClient(IpInferenceClient):
    """Inference Client by name. Client will try to find policy worker configs in the target directory.
    With that said, controller should always setup policy worker and trainer worker before setting up actor
    worker.
    """

    def __init__(self, experiment_name, trial_name, stream_name, rank,
                 serialization_method):
        name = names.inference_stream(experiment_name=experiment_name,
                                      trial_name=trial_name,
                                      stream_name=stream_name)
        server_json_configs = list(sorted(name_resolve.get_subtree(name)))
        target_pw_rank = rank % len(server_json_configs)
        server_config = json.loads(server_json_configs[target_pw_rank])

        super().__init__(server_addresses=server_config['address'].replace(
            "*", "localhost"),
                         serialization_method=serialization_method)
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__stream_name = stream_name

    def get_constant(self, name, timeout=5):
        st = time.time()
        while time.time() - st < timeout:
            try:
                r = name_resolve.get(
                    names.inference_stream_constant(
                        experiment_name=self.__experiment_name,
                        trial_name=self.__trial_name,
                        stream_name=self.__stream_name,
                        constant_name=name)).encode()
                break
            except name_resolve.NameEntryNotFoundError:
                time.sleep(0.1)
        return pickle.loads(binascii.a2b_base64(r))


inference_stream.register_client(
    config_api.InferenceStream.Type.NAME,
    lambda spec: NameResolvingInferenceClient(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
        rank=spec.worker_info.worker_index,
        serialization_method=spec.serialization_method),
)


class NameResolvingInferenceServer(IpInferenceServer):
    """Inference Server By name
    """

    def __init__(self, experiment_name, trial_name, stream_name,
                 serialization_method):
        assert "/" not in experiment_name, "illegal character \"/\" in experiment name"
        assert "/" not in stream_name, "illegal character \"/\" in stream name"

        # Calling super class init will call __make_sockets, which in turn sets values to addresses.
        super().__init__("", serialization_method)
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__stream_name = stream_name
        name_resolve.add_subentry(
            names.inference_stream(experiment_name=experiment_name,
                                   trial_name=trial_name,
                                   stream_name=stream_name),
            json.dumps({
                "address": self._address,
            }),
            keepalive_ttl=15,
        )

    def set_constant(self, name, value):
        """NOTE: Currently set/get constant are implemented for policy state. In other cases, use with caution,
        as values in name_resolve are not guaranteed to be unique or consistent.
        """
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


inference_stream.register_server(
    config_api.InferenceStream.Type.NAME,
    lambda spec: NameResolvingInferenceServer(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
        serialization_method=spec.serialization_method,
    ),
)
