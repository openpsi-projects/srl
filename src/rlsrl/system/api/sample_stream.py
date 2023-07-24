"""This module defines the data flow between the actor workers and the trainers. It is a simple
producer-consumer model.

A side note that our design chooses to let actor workers see all the data, and posts trajectory
samples to the trainer, instead of letting the policy workers doing so.
"""
from typing import Optional, List, Union, Any

import rlsrl.api.config as config_api
import rlsrl.base.buffer as buffer


class NothingToConsume(Exception):
    pass


class SampleProducer:
    """Used by the actor workers to post samples to the trainers.
    """

    def post(self, sample):
        """Post a sample. Implementation should be thread safe.
        Args:
            sample: data to be sent.
        """
        raise NotImplementedError()

    def flush(self):
        """Flush all posted samples.
        Thread-safety:
            The implementation of `flush` is considered thread-unsafe. Therefore, on each producer end, only one
            thread should call flush. At the same time, it is safe to call `post` on other threads.
        """
        raise NotImplementedError()

    def close(self):
        """ Explicitly close sample stream. """
        pass


class SampleConsumer:
    """Used by the trainers to acquire samples.
    """

    def consume_to(self, buffer: buffer.Buffer, max_iter) -> int:
        """Consumes all available samples to a target buffer.

        Returns:
            The count of samples added to the buffer.
        """
        raise NotImplementedError()

    def consume(self) -> Any:
        """Consume one from stream. Blocking consume is not supported as it may cause workers to stuck.
        Returns:
            Whatever is sent by the producer.

        Raises:
            NoSampleException: if nothing can be consumed from sample stream.
        """
        raise NotImplementedError()

    def close(self):
        """ Explicitly close sample stream. """
        pass


class NullSampleProducer(SampleProducer):
    """NullSampleProducer discards all samples.
    """

    def flush(self):
        pass

    def post(self, sample):
        pass


class ZippedSampleProducer(SampleProducer):

    def __init__(self, sample_producers: List[SampleProducer]):
        self.__producers = sample_producers

    def post(self, sample):
        # TODO: With the current implementation, we are pickling samples for multiple times.
        for p in self.__producers:
            p.post(sample)

    def flush(self):
        for p in self.__producers:
            p.flush()


ALL_SAMPLE_PRODUCER_CLS = {}
ALL_SAMPLE_CONSUMER_CLS = {}


def register_producer(type_: config_api.SampleStream.Type, cls):
    ALL_SAMPLE_PRODUCER_CLS[type_] = cls


def register_consumer(type_: config_api.SampleStream.Type, cls):
    ALL_SAMPLE_CONSUMER_CLS[type_] = cls


def make_producer(spec: Union[str, config_api.SampleStream, SampleProducer],
                  worker_info: Optional[config_api.WorkerInformation] = None,
                  *args,
                  **kwargs):
    """Initializes a sample producer (client).

    Args:
        spec: Configuration of the sample stream.
        worker_info: Worker information.
    """
    if isinstance(spec, SampleProducer):
        return spec
    if isinstance(spec, str):
        spec = config_api.SampleStream(type_=config_api.SampleStream.Type.NAME,
                                       stream_name=spec)
    if spec.worker_info is None:
        spec.worker_info = worker_info
    return ALL_SAMPLE_PRODUCER_CLS[spec.type_](spec, *args, **kwargs)


def make_consumer(spec: Union[str, config_api.SampleStream, SampleConsumer],
                  worker_info: Optional[config_api.WorkerInformation] = None,
                  *args,
                  **kwargs):
    """Initializes a sample consumer (server).

    Args:
        spec: Configuration of the sample stream.
        worker_info: Worker information.
    """
    if isinstance(spec, SampleConsumer):
        return spec
    if isinstance(spec, str):
        spec = config_api.SampleStream(type_=config_api.SampleStream.Type.NAME,
                                       stream_name=spec)
    if spec.worker_info is None:
        spec.worker_info = worker_info
    return ALL_SAMPLE_CONSUMER_CLS[spec.type_](spec, *args, **kwargs)


def zip_producers(sample_producers: List[SampleProducer]):
    return ZippedSampleProducer(sample_producers)


register_producer(config_api.SampleStream.Type.NULL,
                  lambda spec: NullSampleProducer)
