"""This module defines the data flow between policy workers and actor workers.

In our design, actor workers are in charge of executing env.step() (typically simulation), while
policy workers running policy.rollout_step() (typically neural network inference). The inference
stream is the abstraction of the data flow between them: the actor workers send environment
observations as requests, and the policy workers return actions as responses, both plus other
additional information.
"""
from typing import List, Optional, Any, Union

import rlsrl.api.config as config_api
import rlsrl.api.policy as policy_api


class InferenceClient:
    """Interface used by the actor workers to obtain actions given current observation."""

    def post_request(self, request: policy_api.RolloutRequest,
                     index: int) -> int:
        """Set the client_id and request_id of the request and cache the request.

        Args:
            request: RolloutRequest of length 1.
            index: Index of the request in the shared memory buffer.
                Used only when using shared memory.
        """
        raise NotImplementedError()

    def poll_responses(self):
        """Poll all responses from inference server.
        This method is considered thread unsafe and
        only called by the main process.
        """
        raise NotImplementedError()

    def is_ready(self, inference_ids: List[int],
                 buffer_indices: List[int]) -> bool:
        """Check whether a specific request is ready to be consumed.

        Args:
            inference_ids: A list of requests to check.
            buffer_indices: The buffer indices of requests to check.

        Outputs:
            is_ready: Whether the inference_ids are all ready.
        """
        raise NotImplementedError()

    def register_agent(self):
        return 0

    def consume_result(self, inference_ids: List[int],
                       buffer_indices: List[int]):
        """Consume a result with specific request_id, returns un-pickled message.
        Raises KeyError if inference id is not ready. Make sure you call is_ready before consuming.

        Args:
            inference_ids: a list of requests to consume.
            buffer_indices: the buffer indices of requests to consume.

        Outputs:
            results: list of rollout_request.
        """
        raise NotImplementedError()

    def flush(self):
        """Send all cached inference requests to inference server.
        Implementations are considered thread-unsafe.
        """
        raise NotImplementedError()

    def get_constant(self, name: str) -> Any:
        """Retrieve the constant value saved by inference server.

        Args:
            name: name of the constant to get.

        Returns:
            value: the value set by inference server.
        """
        raise NotImplementedError()


class InferenceServer:
    """Interface used by the policy workers to serve inference requests."""

    def poll_requests(self) -> List[policy_api.RolloutRequest]:
        """Consumes all incoming requests.

        Returns:
            RequestPool: A list of requests, already batched by client.
        """
        raise NotImplementedError()

    def respond(self, response: policy_api.RolloutResult):
        """Send rollout results to inference clients.

        Args:
            response: rollout result to send.
        """
        raise NotImplementedError()

    def set_constant(self, name: str, value: Any):
        """Retrieve the constant value saved by inference server.

        Args:
            name: name of the constant to get.
            value: the value to be set, can be any object that can be pickled..
        """
        raise NotImplementedError()


ALL_INFERENCE_CLIENT_CLS = {}
ALL_INFERENCE_SERVER_CLS = {}


def register_server(type_: config_api.InferenceStream.Type, cls):
    ALL_INFERENCE_SERVER_CLS[type_] = cls


def register_client(type_: config_api.InferenceStream.Type, cls):
    ALL_INFERENCE_CLIENT_CLS[type_] = cls


def make_server(spec: Union[str, config_api.InferenceStream, InferenceServer],
                worker_info: Optional[config_api.WorkerInformation] = None,
                *args,
                **kwargs):
    """Initializes an inference stream server.

    Args:
        spec: Inference stream specification.
        worker_info: The server worker information.
    """
    if isinstance(spec, InferenceServer):
        return spec
    if isinstance(spec, str):
        spec = config_api.InferenceStream(
            type_=config_api.InferenceStream.Type.NAME,
            stream_name=spec)
    if spec.worker_info is None:
        spec.worker_info = worker_info
    return ALL_INFERENCE_SERVER_CLS[spec.type_](spec, *args, **kwargs)


def make_client(spec: Union[str, config_api.InferenceStream],
                worker_info: Optional[config_api.WorkerInformation] = None,
                *args,
                **kwargs):
    """Initializes an inference stream client.

    Args:
        spec: Inference stream specification.
        worker_info: The client worker information.
    """
    if isinstance(spec, InferenceClient):
        return spec
    if isinstance(spec, str):
        spec = config_api.InferenceStream(
            type_=config_api.InferenceStream.Type.NAME,
            stream_name=spec)
    if spec.worker_info is None:
        spec.worker_info = worker_info
    return ALL_INFERENCE_CLIENT_CLS[spec.type_](spec, *args, **kwargs)
