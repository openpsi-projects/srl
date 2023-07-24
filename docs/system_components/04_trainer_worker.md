# System Components: Trainer Worker

_"Tell me and I forget, teach me and I may remember, involve me and I learn."_

**Trainer Workers** consume training samples and train policy models. Trainer Workers receive training samples from [Actor Workers](02_actor_workers.md) via Sample Consumers. Trainer Workers updates policy models stored in Parameter Database once completing a training step. Although this is a local version of the system, we support multi-GPU training. Multiple trainer workers synchronize their gradients through
[pytorch DistributedDataParallel framework](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).


## class `TrainerWorker`

### Initialization(Configuration)
Refer to the initialization docstring of [TrainerWorker configuration](../user_guide/config_your_experiment.md#TrainerWorker) on how to config a Trainer Worker.

In `_configure()`, steps of initialization are as follows:
1. Initialize buffer, sample consumer and parameter database.
2. Reveal DDP identity (register as a DDP peer in name resolve)
3. Set up DDP: Retrieve DDP peer information from name resolve and initialize DDP connection.
4. Initialize policy and algorithm. Try loading checkpoint for specified policy. If there are no checkpoint, initialize policy model and push the first checkpoint to parameter database.
5. Initialize GPU thread.

### Threaded Implementation 
To optimize the efficiency of GPU usage, there are two threads in the Trainer Worker: main thread and GPU thread. GPU thread runs training tasks. It starts when `start()` is called and stops when `exit()` is called. GPU thread is implemented as a separate class, and details will be introduced in the class documentation.

#### Main Thread
Main threads receives sample batches from sample streams, and controls checkpoint pushing to parameter database. 

The frequency of parameter checking and pushing is controlled by amount of data consumed from sample streams in one `_poll()`. In default setting, in one `_poll`, the main thread consumes 1024 sample batches from sample stream at maximum. 

Tagged version (permanent version) of policy checkpoint is stored in paramter database with a pre-configured frequency. Other versions are stored for backup, but constantly cleared. 

## class `GPUThread`

Class `GPUThread` is a threaded implementation for training with GPU. 

### Initialization

Arguments required for initializing `GPUThread` includes:
1. `buffer`: Buffer instance for storage;
2. `trainer`: Trainer instance for algorithm;
3. `is_master`: Boolean value indicating whether this Trainer Worker is DDP master;
4. `log_frequency_seconds`: Time limit for logging frequency control;
5. `log_frequency_steps`: Step limit for logging frequency control;
6. `push_frequency_seconds`: Time limit for checkpoint pushing frequency control;
7. `push_frequency_steps`: Step limit for checkpoint pushing frequency control;
8. `dist_kwargs`: DDP related arguments.

See [Basic Utils](09_basic_utils.md) for more details about frequency control.

### Main Loop

GPU thread runs in a dead loop until interruption. One step in the loop includes:
1. Check whether buffer is empty. If yes, sleep for 5 ms to void locking the buffer. (TODO: implement thread-safe buffer) If no, get an replay entry.
2. Run one training step on replay entry.
3. Check logging condition and store log into queue.
4. Check pushing checkpoint condition and store checkpoint into queue.

### Logging

When `TrainerWorker._stats()`  is called, if the trainer is DDP master, `GPUThread.stats()` will be called to return all logged stats in logging queue.

### Terminating 

When Trainer Worker is exiting, it will tell `GPUThread` when to stop running. To prevent DDP peers from waiting for other exited peers, DDP peers should stop training with only a few step differences. Trainer Worker will calculate when to stop GPU thread safely and tell GPU thread to enter an interrupt loop.

# Related References

- [System Overview](01_system_overview.md)
- [Actor Worker](02_actor_worker.md)
- [Basic Utils](09_basic_utils.md)

# Related Files and Directories

- [system/impl/trainer_worker.py](../../src/rlsrl/system/impl/trainer_worker.py)

# What's Next

- [Buffer Worker](05_buffer_worker.md)