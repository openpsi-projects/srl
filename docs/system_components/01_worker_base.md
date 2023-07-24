# System Components: Worker Base 
### Base class for workers: `worker_base.Worker`

Base class of all workers. Common handles implemented in class `worker_base.Worker` are:

- configure
- start
- pause
- exit
- run

Upon launching the system, a centralized controller will launch the workers (in this local version, launching method is python `multiprocessing`) and call these handles to control the life cycle of workers.

### Worker Status

There are two status for a live worker, running and exiting. Workers could be neither running nor exiting when they are not configured or paused. There are three handles that only changes worker status by default. They are:

- `start`: change status running to true.
- `pause`: change status running to false.
- `exit`: change status exiting to true

### Life Cycle

In local version, workers' life cycles are simplified. Three methods of workers are called after launching workers and before exiting or any pausing: `configure()`, `start()`, and `run()`. In distributed version, other handles are used by controller to provide functions such as reconfiguring, pausing and monitoring.  

When `configure()` is called, `_configure()` method implemented in subclasses will run to parse worker specific configs and initializations. After that, worker will run common configs such as `wandb` logging configuration.

When `run()` is called, worker will run in a dead loop of `_poll()` if `start()` is called to set the worker running. After every `_poll()` the worker will log the default stats and worker stats specified in `_stats()` to standard output or/and `wandb`. 

### Implement subclasses of `worker_base.Worker`

If you want to implement some type of Worker on your own, you should **at least** override method `_configure()` and `_poll()`. 

In `_configure()`, you may initialize your worker by completing following steps:
1. Configure required parameters and variables (by reading them from configuration file).
2. Initialize I/O, storage or computation components, assign devices and computing resources.
3. Initialize and start threads if your worker is implemented in a threaded fashion. Details will be discussed in class `MappingThread`.
4. Configure worker specific monitoring.

In `_poll()`, you should implement the main computing step of your worker. The return value (sample counts and batch counts) could be arbitrarily defined to whatever you want to log. 

In addition, you could override handles that changes worker status: `pause()`, `start()` and `exit()` to change behavior of workers upon status change. 

If you desire to record extra information when running worker, specify them in `_stats()` to log then down in standard output or `wandb`.

### Scheduling
The local version of the system are scheduled with default python `multiprocessing` rules. Workers are launched as `multiprocessing.Process` and connected by data streams implemented in `multiprocessing.Queue`. In the distributed version, workers are scheduled via [slurm](https://slurm.schedmd.com/documentation.html) scheduler. 


## class `MappingThread`

The workers could be implemented in a threaded fashion using python `threading` module. For example, in our original workers:
- Policy Worker has a main thread(cpu), an inference thread, and a responding thread(cpu).
- Trainer Worker has a main thread(cpu, where buffer resides), and a training thread(gpu).

`Mapping Thread` is a wrapper of python `threading.Thread` that makes it easier to implement a threaded worker. A mapping thread gets input from its upstream queue, process data and outputs into a downstream queue.

### Initialization

`Mapping Thread` is initialized with:
1. `map_fn`: A mapping function that takes input and process it to get output.
2. `upstream_queue`: Input queue.
3. `downstream_queue`: Optional. Output queue, should be none if there is no output for `map_fn`.
4. `cuda_device`: Optional. CUDA device that should be used in this mapping thread.

### Handles

There are 4 handles for an instance of `MappingThread`, similar to python `threading.Thread`:
1. `is_alive()`: check whether the thread is alive.
2. `start()`: Start the thread. After starting, the thread will take input from `upstream_queue`, call `map_fn` and outputs into `downstream_queue` repeatedly.
3. `join()`: Join the thread.
4. `stop()`: Stop the thread. 

# Related References

- [System Overview](00_system_overview.md)

# Related File and Directory
- [system/api/worker_base.py](../../src/rlsrl/system/api/worker_base.py)

# What's Next

- [Actor Worker](02_actor_worker.md)

