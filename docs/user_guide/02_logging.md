# User Guide: Logging

There are two ways of logging we support in our system: standard output and [wandb](https://wandb.ai/site). In the system, the only statistic logging is done in **workers**, which is implemented in the main logic of class `WorkerBase`. 

## Standard Output Logging

To enable standard output logging for some worker, set `log_terminal` in `WorkerInformation` spec of this worker to true.

## wandb Logging

To enable standard output logging for some worker, set `log_wandb` in `WorkerInformation` spec of this worker to true, and complete following fields:

- `wandb_entity`
- `wandb_project`
- `wandb_job_type`
- `wandb_group`
- `wandb_name`

By default, these values will be setup by the system with the following defaults:
- `wandb_entity = None`
- `wandb_project = experiment_name` (-e)
- `wandb_group = trial_name` (-f)
- `wandb_job_type = worker_type` (actor/policy/trainer/eval_manager)
- `wandb_name = policy_name or "unnamed" `

The worker configuration will also be passed as argument ```config``` to ```wandb.init()```. Nested dataclasses are not parsed by W&B. For example, currently trainer configuration in `TrainerWorker` cannot be used as filter. This is a 
known issue and will be resolved in the future.  A workaround is to add the values that you want to filter on to `wandb_name`. See below for configuration instructions.

You may specify your customized configuration in you experiment configuration. An example will be:

```python

from rlsrl.api.config import *

actor_worker_count = 10
actor_worker_configs = [experiments.ActorWorker(...,  # worker configuration
                                                worker_info=experiments.WorkerInformation(
                                                    wandb_entity="your_entity",
                                                    wandb_project="your_project",
                                                    wandb_group="your_group",
                                                    wandb_job_type="actor_worker",
                                                    wandb_name=f"my_perfect_wandb_name_actor_{worker_index}")
                                               ) for worker_index in range(actor_worker_count)]
```

## Logging Arbitrary Data

When running your own experiments, you may want to log data that is not explicitly shown in our orignial implementation. In this situation, you will have to understand how logging in workers works and re-implement logging function `_stats` in workers (reference: [System Components](01_worker_base.md)). 

Everytime a worker finishes a round of `_poll()`, it will try to call `_stats()` function to retrive any data that is required to be logged and log them by `logger` (terminal logging) or `self.__wandb_run.log(...)` (wandb logging) to complete a logging step. `_stats()` function returns a list of `LogEntry` that stores data. A `LogEntry` contains 2 fields, a stats dict `stats` and `step` (used for wandb logging by steps). On every logging step, log entries in list returned by `_stats()` will be logged one by one. For a log entry `e`, if terminal logging is used, `e.stats` will be printed. If wandb logging is used and `e.step >= 0`, `self.__wandb_run.log(e.stats, step=e.step)` will be called. If `e.step < 0` (default), `self.__wandb_run.log(e.stats)` will be called.

If you wish to log your own data, you need to modify source code for worker implementations. Store data as attributes of worker class and return them as log entries in `_stats()`.

# Related References
- [System Components: Worker Base](../01_worker_base.md)

# Related Files and Directories
- [system/worker_base.py](../../src/rlsrl/system/worker_base.py)

# What's next
- [User Guide: Environments](03_environments.md)
