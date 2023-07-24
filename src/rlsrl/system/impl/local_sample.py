import logging
import numpy as np
import time
import threading

import rlsrl.api.config as config_api
import rlsrl.base.namedarray as namedarray
import rlsrl.base.shared_memory as shared_memory
import rlsrl.system.api.sample_stream as sample_stream

logger = logging.getLogger("LocalSampleStream")


class SharedMemorySampleProducer(sample_stream.SampleProducer):

    def __init__(
        self,
        experiment_name,
        trial_name,
        stream_name,
        qsize,
        ctrl: shared_memory.OutOfOrderSharedMemoryControl,
    ):
        self.__shared_memory_writer = shared_memory.SharedMemoryDock(
            experiment_name,
            trial_name,
            stream_name + "_sample",
            qsize=qsize,
            ctrl=ctrl,
            second_dim_index=True,
        )
        self.__post_lock = threading.Lock()
        self.__sample_buffer = []

    def post(self, sample):
        with self.__post_lock:
            self.__sample_buffer.append(sample)

    def flush(self):
        with self.__post_lock:
            tmp = self.__sample_buffer
            self.__sample_buffer = []
        for x in tmp:
            self.__shared_memory_writer.write(x)

    def close(self):
        self.__shared_memory_writer.close()


sample_stream.register_producer(
    config_api.SampleStream.Type.SHARED_MEMORY,
    lambda spec, ctrl: SharedMemorySampleProducer(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
        qsize=spec.qsize,
        ctrl=ctrl,
    ),
)


class SharedMemorySampleConsumer(sample_stream.SampleConsumer):

    def __init__(
        self,
        experiment_name,
        trial_name,
        stream_name,
        qsize,
        batch_size,
        ctrl: shared_memory.OutOfOrderSharedMemoryControl,
    ):
        self.__shared_memory_reader = shared_memory.SharedMemoryDock(
            experiment_name,
            trial_name,
            stream_name + "_sample",
            qsize=qsize,
            ctrl=ctrl,
            second_dim_index=True)
        self.__batch_size = batch_size

    def consume_to(self, buffer, max_iter=16):
        count = 0
        for _ in range(max_iter):
            try:
                sample = self.__shared_memory_reader.read(
                    batch_size=self.__batch_size)
            except shared_memory.NothingToRead:
                break
            if_batch = buffer.put(sample)
            count += 1
        return count

    def consume(self):
        try:
            return self.__shared_memory_reader.read(
                batch_size=self.__batch_size)
        except shared_memory.NothingToRead:
            raise sample_stream.NothingToConsume()

    def close(self):
        self.__shared_memory_reader.close()


sample_stream.register_consumer(
    config_api.SampleStream.Type.SHARED_MEMORY,
    lambda spec, ctrl: SharedMemorySampleConsumer(
        experiment_name=spec.worker_info.experiment_name,
        trial_name=spec.worker_info.trial_name,
        stream_name=spec.stream_name,
        qsize=spec.qsize,
        batch_size=spec.batch_size,
        ctrl=ctrl,
    ),
)


class InlineSampleProducer(sample_stream.SampleProducer):
    """Testing Only! Will not push parameters.
    """

    def __init__(self, trainer, policy):
        from rlsrl.api.trainer import make
        from rlsrl.system.api.parameter_db import make_db
        from rlsrl.api.config import ParameterDB

        self.trainer = make(trainer, policy)
        self.buffer = []
        self.logger = logging.getLogger("Inline Training")
        self.param_db = make_db(
            ParameterDB(type_=ParameterDB.Type.LOCAL_TESTING))
        self.param_db.push(name="",
                           checkpoint=self.trainer.get_checkpoint(),
                           version=0)

    def post(self, sample):
        self.buffer.append(sample)
        self.logger.debug("Receive sample.")

    def flush(self):
        if len(self.buffer) >= 5:
            batch_sample = namedarray.recursive_aggregate(
                self.buffer, aggregate_fn=lambda x: np.stack(x, axis=1))
            batch_sample.policy_name = None
            self.trainer.step(batch_sample)
            self.param_db.push(name="",
                               checkpoint=self.trainer.get_checkpoint(),
                               version=0)
            self.logger.info("Trainer step is successful!")
            self.logger.debug(
                f"Trainer steps. now on version {self.trainer.policy.version}."
            )
            self.buffer = []


sample_stream.register_producer(
    config_api.SampleStream.Type.INLINE_TESTING,
    lambda spec: InlineSampleProducer(
        trainer=spec.trainer,
        policy=spec.policy,
    ),
)
