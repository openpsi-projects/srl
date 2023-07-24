"""This module defines the data flow between the actor workers and the trainers. It is a simple
producer-consumer model.

A side note that our design chooses to let actor workers see all the data, and posts trajectory
samples to the trainer, instead of letting the policy workers doing so.
"""
from typing import Optional, List, Union, Any
import threading
import logging
import json
import numpy as np
import pickle
import socket
import warnings
import zmq
import os
import time
from statistics import mean

from rlsrl.api.trainer import SampleBatch
import rlsrl.api.config as config_api
import rlsrl.base.buffer as buffer
import rlsrl.base.name_resolve as name_resolve
import rlsrl.base.namedarray as namedarray
import rlsrl.base.names as names
import rlsrl.system.api.sample_stream as sample_stream

logger = logging.getLogger("RemoteSampleStream")


class IpSampleProducer(sample_stream.SampleProducer):
    """A simple implementation: sends all samples to a specific consumer (trainer worker).
    """

    def __init__(self, target_address, serialization_method):
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.PUSH)
        self.__socket.connect(f"tcp://{target_address}")
        self.__socket.setsockopt(zmq.LINGER, 0)
        self.__sample_buffer = []
        self.__post_lock = threading.Lock()
        self.__serialization_method = serialization_method

    def __del__(self):
        self.__socket.close()

    def post(self, sample):
        data = namedarray.dumps(sample, method=self.__serialization_method)
        with self.__post_lock:
            self.__sample_buffer.append(data)

    def flush(self):
        with self.__post_lock:
            ds = self.__sample_buffer
            self.__sample_buffer = []
        for d in ds:
            self.__socket.send_multipart(d)


sample_stream.register_producer(
    config_api.SampleStream.Type.IP,
    lambda spec: IpSampleProducer(
        target_address=spec.address,
        serialization_method=spec.serialization_method,
    ),
)


class IpSampleConsumer(sample_stream.SampleConsumer):

    def __init__(self, address):
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.PULL)
        self.__socket.RCVTIMEO = 200
        self.__socket.setsockopt(zmq.LINGER, 0)
        self.__socket.setsockopt(zmq.RCVHWM, 10)
        if address == "":
            host_ip = socket.gethostbyname(socket.gethostname())
            port = self.__socket.bind_to_random_port(f"tcp://{host_ip}")
            address = f"{host_ip}:{port}"
        else:
            self.__socket.bind(f"tcp://{address}")
        self.address = address
        self.first_time = True

    def __del__(self):
        self.__socket.close()

    def consume_to(self, buffer, max_iter=16):
        count = 0
        for _ in range(max_iter):
            try:
                data = self.__socket.recv_multipart(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            try:
                sample = namedarray.loads(data)
            except pickle.UnpicklingError:
                warnings.warn("Unpickling failed. One sample is discarded.")
                continue
            buffer.put(sample)
            if self.first_time:
                logger.info(f"one sample size {namedarray.size_bytes(sample)}")
                self.first_time = False
            count += 1
        return count

    def consume(self) -> Any:
        """Note that this method blocks for 0.2 seconds if no sample can be consumed. Therefore, it is safe to make
        a no-sleeping loop on this method. For example:
        while not interrupted:
            try:
                data = consumer.consume()
            except NothingToConsume:
                continue
            process(data)
        """
        try:
            data = self.__socket.recv_multipart()
        except zmq.ZMQError:
            raise sample_stream.NothingToConsume()
        try:
            sample = namedarray.loads(data)
        except pickle.UnpicklingError:
            warnings.warn("Unpickling failed. One sample is discarded.")
            raise sample_stream.NothingToConsume()
        return sample


sample_stream.register_consumer(
    config_api.SampleStream.Type.IP,
    lambda spec: IpSampleConsumer(address=spec.address, ),
)


class NameResolvingSampleProducer(IpSampleProducer):

    def __init__(self, experiment_name, trial_name, stream_name, rank,
                 serialization_method):
        name = names.sample_stream(experiment_name=experiment_name,
                                   trial_name=trial_name,
                                   stream_name=stream_name)
        server_json_configs = list(sorted(name_resolve.get_subtree(name)))

        assert len(server_json_configs) > 0, f"No trainer configuration found. Initialize trainers first" \
                                             f"experiment: {experiment_name} stream_name: {stream_name}"
        target_tw_rank = rank % len(server_json_configs)
        server_config = json.loads(server_json_configs[target_tw_rank])

        super().__init__(target_address=server_config['address'].replace(
            "*", "localhost"),
                         serialization_method=serialization_method)


sample_stream.register_producer(
    config_api.SampleStream.Type.NAME,
    lambda spec: NameResolvingSampleProducer(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
        rank=spec.worker_info.worker_index,
        serialization_method=spec.serialization_method,
    ),
)


class NameResolvingSampleConsumer(IpSampleConsumer):

    def __init__(self, experiment_name, trial_name, stream_name, address=""):
        super().__init__(address)
        self.__name_entry = name_resolve.add_subentry(
            names.sample_stream(experiment_name=experiment_name,
                                trial_name=trial_name,
                                stream_name=stream_name),
            json.dumps({"address": self.address}),
            keepalive_ttl=15)


sample_stream.register_consumer(
    config_api.SampleStream.Type.NAME,
    lambda spec: NameResolvingSampleConsumer(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
    ),
)


class NameResolvingMultiAgentSampleProducer(NameResolvingSampleProducer):

    def __init__(self, experiment_name, trial_name, stream_name, rank,
                 serialization_method):
        super(NameResolvingMultiAgentSampleProducer,
              self).__init__(experiment_name, trial_name, stream_name, rank,
                             serialization_method)
        self.__cache = []
        self.logger = logging.getLogger("MA Producer")

    def post(self, sample: SampleBatch):
        self.__cache.append(sample)

    def flush(self):
        if self.__cache:
            if None in [
                    sample.unique_policy_version for sample in self.__cache
            ]:
                self.__cache = []
                return

            super(NameResolvingMultiAgentSampleProducer, self).post(
                namedarray.recursive_aggregate(self.__cache,
                                               lambda x: np.stack(x, axis=1)))
            self.logger.debug(
                f"posted samples with "
                f"version {[sample.unique_policy_version for sample in self.__cache]} "
                f"and name {[sample.unique_policy_name for sample in self.__cache]}."
            )
            self.__cache = []


sample_stream.register_producer(
    config_api.SampleStream.Type.NAME_MULTI_AGENT,
    lambda spec: NameResolvingMultiAgentSampleProducer(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
        rank=spec.worker_info.worker_index,
        serialization_method=spec.serialization_method,
    ),
)


class RoundRobinNameResolvingSampleProducer(sample_stream.SampleProducer):

    def __init__(self, experiment_name, trial_name, stream_name,
                 serialization_method):
        name = names.sample_stream(experiment_name=experiment_name,
                                   trial_name=trial_name,
                                   stream_name=stream_name)
        server_json_configs = list(sorted(name_resolve.get_subtree(name)))

        assert len(server_json_configs) > 0, f"No trainer configuration found. Initialize trainers first" \
                                             f"experiment: {experiment_name} stream_name: {stream_name}"
        self.__streams = [
            NameResolvingSampleProducer(
                experiment_name,
                trial_name,
                stream_name,
                rank=r,
                serialization_method=serialization_method)
            for r in range(len(server_json_configs))
        ]
        self.__current_idx = 0

    def post(self, sample):
        self.__streams[self.__current_idx].post(sample)
        self.__current_idx = (self.__current_idx + 1) % len(self.__streams)

    def flush(self):
        for p in self.__streams:
            p.flush()


sample_stream.register_producer(
    config_api.SampleStream.Type.NAME_ROUND_ROBIN,
    lambda spec: RoundRobinNameResolvingSampleProducer(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
        serialization_method=spec.serialization_method,
    ),
)


class BroadcastNameResolvingSampleProducer(sample_stream.SampleProducer):

    def __init__(self, experiment_name, trial_name, stream_name,
                 serialization_method):
        name = names.sample_stream(experiment_name=experiment_name,
                                   trial_name=trial_name,
                                   stream_name=stream_name)
        server_json_configs = list(sorted(name_resolve.get_subtree(name)))

        assert len(server_json_configs) > 0, f"No trainer configuration found. Initialize trainers first" \
                                             f"experiment: {experiment_name} stream_name: {stream_name}"
        self.__streams = [
            NameResolvingSampleProducer(
                experiment_name,
                trial_name,
                stream_name,
                rank=r,
                serialization_method=serialization_method)
            for r in range(len(server_json_configs))
        ]

    def post(self, sample):
        for stream in self.__streams:
            stream.post(sample)

    def flush(self):
        for stream in self.__streams:
            stream.flush()


sample_stream.register_producer(
    config_api.SampleStream.Type.NAME_BROADCAST,
    lambda spec: BroadcastNameResolvingSampleProducer(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
        serialization_method=spec.serialization_method,
    ),
)
