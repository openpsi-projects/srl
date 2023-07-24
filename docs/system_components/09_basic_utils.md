# System Components: Basic Utils

In this section, basic utilities including parameter database in the system will be introduced in detail.

## Parameter Server (class `ParameterDBClient`)

`ParameterDBClient` provides communication between an user and the parameter database (aka parameter server). For details about parameters (Policy models) naming rules and handles in parameter database client, see [system/parameter_db.py](../../src/rlsrl/system/parameter_db.py).

### Implementation: Pytorch Filesystem Parameter Database (class `PytorchFilesystemParameterDB(ParameterDBClient)`)

An implementation of parameter DB that stores `pytorch` models in the filesystem. All files are stored in `"$HOME/marl_checkpoints"` by default. If you wish to change the directory, change `ROOT` parameter under this class. 

## Buffer (class `Buffer`)

A buffer that stores training sample, which is used in [Trainer Worker](04_trainer_worker.md). It has 3 simple methods:

- `put(x)`: put `x` into buffer storage.
- `get()`: get next element.
- `empty()`: check if the buffer is empty.

Related file [base/buffer.py](../../src/rlsrl/base/buffer.py)

### Implementation

In all following implementations, buffers store samples in a unit of batch, which includes`batch_size` samples. One batch of samples is stored in a `ReplayEntry` data structure, which has fields :`reuses_left`, `receive_time`, `sample` and `reuses`. 

#### Simple Queue Buffer (class `SimpleQueueBuffer(Buffer)`)

A simple buffer that is implemented with a python `queue.SimpleQueue()`, following FIFO pattern.

#### Simple Replay Buffer (class `SimpleReplayBuffer(Buffer)`)

A buffer that allows to get one sample batch for a pre-configured number of times. It uniformly samples a sample batch when calling `get()`. When sample batch reaches maximum replay time, it is discarded.

#### Priority Buffer (class `PriorityBuffer(Buffer)`)

A replay buffer that get batches with the maximal `reuses_left`. 

## GPU utils 

GPU utils, related file: [base/gpu_utils.py](../../src/rlsrl/base/utils.py).

## Named Array (class `NamedArray`)

**Named Array** is a data structure that is used everywhere in the system. A class modified from the `namedarraytuple` class in rlpyt repo, referring to https://github.com/astooke/rlpyt/blob/master/rlpyt/utils/collections.py#L16.

NamedArray supports dict-like unpacking and string indexing, and exposes integer slicing reads and writes applied to all contained objects, which must share indexing (`__getitem__`) behavior (e.g. numpy arrays or torch tensors).

Note that namedarray supports nested structure, i.e., the elements of a NamedArray could also be NamedArray. 

See related file [base/namedarray.py](../../src/rlsrl/base/namedarray.py) for implementation details, and read [User Guide: NamedArray](../user_guide/06_named_array.md) for usage guide and examples.

## Names

Methods to get names to store in name resolving, related file: [base/names.py](../../src/rlsrl/base/names.py).

## Name Record Repository (class `NameRecordRepository`)

Name record repository, also referred as **name resolve**, implements a simple name resolving service, which can be considered as a global key-value dict. See related file [base/name_resolve.py](../../src/rlsrl/base/name_resolve.py) for detailed info about APIs.

## Network

Utility functions about networking, related file [base/network.py](../../src/rlsrl/base/network.py)

## Numpy Utils

Utility functions about numpy, related file [base/numpy.py](../../src/rlsrl/base/numpy.py)

## Time Utils (class `FrequencyControl`)

Frequency Control is an utility to control the execution of code with a time or/and step frequency, used when workers needs a timing method to control frequency of some operations. See file [base/timeutil.py](../../src/rlsrl/base/util.py) for detailed usage. 

## User Utils

Utility functions about OS and users, related file [base/user.py](../../src/rlsrl/base/user.py)


# Related References
- [System Overview](00_system_overview.md)

# Related Files and Directories
- [system/api/parameter_db.py](../../src/rlsrl/system/api/parameter_db.py)
- [base/](../../src/rlsrl/base/)
