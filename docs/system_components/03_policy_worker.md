# System Components: Policy Worker

_"The best policy is to declare victory and leave."_

A **Policy Worker** does policy model inference to produce actions as inputs for environment simulations in [Actor Workers](02_actor_worker.md). Policy Workers receives inference requests from _Inference Servers_, finish policy inference designated by the requests, and send the results to [Actor Workers](02_actor_worker.md) via _Inference Servers_. Policy Workers are also required to update their policy model in a required frequency via _Parameter Database_. (_Inference Server_ and _Parameter Database_ are introduced in [System Overview](01_system_overview.md))

## class `PolicyWorker`
### Initialization (Configuration)
Refer to the initialization docstring of [PolicyWorker configuration](../user_guide/config_your_experiment.md#PolicyWorker) on how to config a Policy Worker.

There are 3 steps in configuration (in function `_configure()`):
1. Make policy model for inference.
2. Initialize Inference Server and parameter database that stores policy model.
3. Initialize and start rollout thread and respond thread.

### Threaded Implementation
Policy Worker is implemented in a threaded fashion to optimize efficiency. There are three threads for a policy worker, which will be introduced in detail in the following sections.

#### Main Thread
The main thread is responsible for two tasks (in function`_poll()`):
1. Pulling parameters from database or files. Main thread actively get policy checkpoint from database or files on a pre-configured frequency, then put the parameters into parameter queue for inference thread to update the policy.
2. Receiving and batching rollout requests. The main thread receives inference requests from Inference Stream, batch all unprocessed inference requests into inference queue (for inference thread to process) until the batch reaches a pre-configured batch size. In the case that number of unprocessed inference requests exceeds batch size, the main thread puts them into a request buffer and wait for inference of prior batch. 

Notice that rollout batch size is dynamically adjusted according to the throughput of inference requests. The inference queue is a queue of size 1, and when the queue is full, the main thread accumulates requests. Once the inference queue is cleared, the main-thread batches all the pending requests and put to the queue.

The batch count of policy worker is number batches that the worker has processed, and sample count is number of inference requests the worker has processed.

#### Rollout Thread
The rollout thread updates policy model from parameter queue, and runs `policy.rollout` on request batch in inference queue on every step. After rollout, it puts the results into respond queue.  

#### Respond Thread
The respond queue sends the inference results (from respond queue) to the Inference Stream. 


# Related References

- [System Overview](01_system_overview.md)
- [Actor Worker](02_actor_worker.md)

# Related Files and Directories

- [system/impl/policy_worker.py](../../src/rlsrl/system/impl/policy_worker.py)

# What's Next

- [Trainer Worker](04_trainer_worker.md)
