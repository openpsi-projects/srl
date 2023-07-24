# System Components: Sample Stream

**Sample Stream** defines the data flow between the actor workers and the trainers. It is a simple producer-consumer model.

A side note that our design chooses to let actor workers see all the data, and posts trajectory samples to the trainer, instead of letting the policy workers doing so.

## class `SampleProducer`

Interface used by the actor workers to post samples to the trainers.

See [system/sample_stream.py](../../src/rlsrl/system/api/sample_stream.py) for detailed API description.

## class `SampleConsumer`

Interface used by the trainers to acquire samples.

See [system/sample_stream.py](../../src/rlsrl/system/sample_stream.py) for detailed API description.

## class `ZippedSampleProducer(SampleProducer)`

Sometimes one copy of training sample are required to be sent to multiple consumers. `ZippedSampleProducer` is a set of multiple sample producers. When `ZippedSampleProducer.post(sample)` is called, `sample` is sent by all sample producers. 

## Implementations

### Shared Memory Local Sample Stream (class `SharedMemorySampleProducer`, `SharedMemorySampleConsumer`)

A sample stream implementation with python shared memory.

### Shared Memory Sample Stream (class `IpSampleProducer`, `IpSampleConsumer`)

A sample stream implementation with sockets.

### Shared Memory Sample Stream (class `NameResolvingSampleProducer`, `NameResolvingSampleConsumer`)

Sample Stream implementation with name resolveing service to match producers and consumers.

### Null Sample Producer (class `NullSampleProducer`)

A dummy sample producer that discard all samples. 


# Related References
- [System Overview](00_system_overview.md)
- [Actor Worker](02_actor_worker.md)
- [Trainer Worker](04_trainer_worker.md)

# Related Files and Directories
- [system/api/sample_stream.py](../../src/rlsrl/system/api/sample_stream.py)
- [system/impl/local_sample.py](../../src/rlsrl/system/impl/local_sample.py)
- [system/impl/remote_sample.py](../../src/rlsrl/system/impl/remote_sample.py)

# What's next

- [Basic Utils](09_basic_utils.md)