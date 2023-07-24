# User Guide: Experiment Config

Classes and methods appear in this section are implemented in [api/config.py](../../rlsrl/api/config.py) by default. [api/config.py](../../rlsrl/api/config.py) includes specifications for experiment configuration.

To understand experiment configuration in the system, two concepts are important: **Experiments** and **Trials**. An **Experiment** is defined by a subclass of `Experiment`, including basic information about the environment, algorithm, policy, and workers. To run a specific experiment, the experiment should be registered with method `register_experiment(name, *class)`. A **Trial** is a running instance of an **Experiment**. Different **Trials** in an **Experiment** usually has different parameters or settings, but shares the same type of algorithm, policy and enviroment.

For example, you have implemented a experiment config file `my_config.py` in some directory, in which a class `MyExperiment(Experiment)` is defined. `MyExperiment` class should implement a `initial_setup()` method that returns `ExperimentConfig` that includes worker configurations:

```python
# my_config.py
from rlsrl.api.config import *

class MyExperiment(Experiment):
    def initial_setup(self):
        return ExperimentConfig(
            actor_workers = ..., # list of ActorWorker instances, actor worker specifications
            policy_workers = ..., # list of PolicyWorker instances, policy worker specifications
            trainer_workers = ..., # list of TrainerWorker instances, trainer worker specifications
            ...
        )
```

To register the experiment, you should write:

```python
register_experiment(my_expr, MyExperiment)
```

at the end of `my_config.py`. To run a trial of `MyExperiment`, you should use the experiment name you registered, specify a trial name, as well as import `my_config.py`, by running:

`srl-local -e my_expr -f some_trial --import_file my_config.py`

Upon running, `make_experiment(name)` method will pass all configurations to main process for building and configuring workers.

We use **specifications** (or **spec**) to configure components in the system. In implementation, one specification is a nested dataclass that includes all the information to construct corresponding class instance. Here, we introduce `ActorWorker` specification as an example.

## Specs

### Actor Worker

To config [actor workers](../02_actor_worker.md), following fields should be specified for `ActorWorker`:
- `env`: class `Environment` or a string that can be converted to such a class, an environment specification nested in `ActorWorker` see [User Guide: Environments](03_environments.md).
- `sample_streams`: list of `SampleStream` or strings that can be converted to such a class, specifications of sample streams connected to this actor worker.
- `inference_streams`: list of `InferenceStream` or strings that can be converted to such a class, specifications of inference streams connected to this actor worker.
- `agent_specs`: list of `AgentSpec`, specifications of agents that act in the environment of this actor worker.
- `max_num_steps`: maximum number of steps to run for an agent.
- `ring_size`: number of environments in an EnvRing.
- `inference_splits`:  actor worker will split the ring into `inference_splits` parts and flush the inference clients for each part.
- `worker_info`: class `WorkerInformation`, specification for common worker information, which should be in all workers.

As shown above, `env`, `sample_streams`, `inference_streams` and `agent_specs` are nested specifications in `ActorWorker` that configure sub-components of an actor worker. In the following part, all specs available in the system will be listed.

### WorkerInformation

Basic information for a worker, exists in all worker specs. Followings are fields that should not be filled by users in experiment config. The system will automatically fill these fields by reading other configurations and options:

- `experiment_name`
- `trial_name`
- `worker_type`
- `worker_index`
- `worker_count`
- `worker_tag`
- `policy_name`

Following fields should be specified by users if needed:

- wandb logging information (see [User Guide: Logging](02_logging.md))
    - `log_wandb`: whether to log this worker in wandb.
    - `wandb_entity`
    - `wandb_project`
    - `wandb_job_type`
    - `wandb_group`
    - `wandb_name`
- `log_terminal`: whether to log in terminal.
- `device`: `cpu` or digit that indicate GPU device (for example `0` when CUDA_VISIBLE_DEVICES="0,1" to use GPU 0 for this worker.)

### Environment

To config an [environment](03_environments.md), you should specify:

- `type_`: a string, name of a registered environment.
- `args`: a dict of arguments that used to initialize the envrionment.

### AgentSpec

Spec for an agent in environments in an actor worker. `ActorWorker` should contain a list of `AgentSpec` to configure all agents in the environments. Note that all environments copies in actor worker are the same and shares agent specs. To config an agent (../02_actor_worker.md), you should specify: 

- `index_regex`: regular expression string that matches agent indices (which are natural numbers)
- `inference_stream_idx`: integer, position of inference stream used in inference stream list in `ActorWorker` spec.
- `sample_stream_idx`: integer, position of sample stream used in sample stream list in `ActorWorker` spec.
- `sample_steps`: integer number of steps in one training sample. Effective only when `send_full_trajectory` is false.
- `bootstrap_steps`: integer number of bootstrap steps. Temporal-difference style value tracing benefits from bootstrapping.
- `deterministic_action`: whether to use deterministic action, usually false for training, true for evaluation.
- `send_after_done`: whether to only send samples after the environment is done.
- `send_full_trajectory`: whether to send full trajectory of an episode to sample streams. Mostly used when episodes are of fixed length, or when agent is running evaluation.
- `pad_trajectory`: whether to pad trajectory to a fixed length when `send_full_trajectory=True`.
- `send_concise_info`: if set to true, each episode is contracted in one time step, with the first observation and the last episode-info.
- `stack_frames`: integer, 0: raw stacking; 1: add a new axis on dimension 0; >=2: stack n frames on axis 0.


### Policy Worker

To configure a [policy worker](../03_policy_worker.md), the following fields must be specified:

- `policy_name`: string name of a policy instance, an identifier for workers to communicate. One policy worker should serve only one policy instance. If you are training two policies named "alice" and "bob" in the same experiment, they won't interfere with each other. You are free to choose your `policy_name(s)` in you experiment.
- `inference_stream`: `InferenceStream` spec, one and only one inference stream should be connected to one policy worker.
- `policy`: `Policy` spec.
- `parameter_db`: `ParameterDB` spec, note that type of parameter databases in workers should be the same in one experiment.
- `batch_size`: maximum batch size for inference. Since policy worker dynamically batch the requests, `batch_size` should be as large as possible until falling short of GPU memory.
- `policy_identifier`: string that identify which policy version the policy worker should load when starting.
- `pull_frequency_seconds`: minimum interval time length between two policy pulls.
- `foreign_policy`: `ForeignPolicy` spec, if not none, identify a foreign policy (not from this experiment) to load.
- `worker_info`: `WorkerInformation` spec.

### Policy

Spec for a policy. See [User Guide: Policies and algorithms](04_policy_algorithm.md) for definition of a policy. Fields:

- `type_`: type of the policy
- `args`: argument that should be specified when constructing a policy instance.

### ParameterDB

Parameter database spec, only support filesystem type in local version:

- `type_`: type of paramter DB, should be `ParameterDB.Type.FILESYSTEM` only.

### PolicyInfo

A spec that provides all the information to pull a policy model from storage. Fields:

- `worker_info`: worker information that used to specify the storage path of the policy model. Note that the worker could be from a different trial or experiment.
- `policy_name`: same as in `PolicyWorker`.
- `policy_identifier`: same as in `PolicyWorker`.
- `absolute_path`: absolute path to read policy instance from. When this field is specified, other fields will be ignored.
- `param_db`: type of parameter database to construct when loading the policy.

### ForeignPolicy

A spec that identify a policy that is not from this trial (or/and experiment). Used for workers to read a foreign policy. Following fields should be specified when identifying a foreign policy:

- `foreign_experiment_name`
- `foreign_trial_name`
- `foreign_policy_name`
- `foreign_policy_identifier`
- `absolute_path`: absolute path to read foreign policy instance from. When this field is specified, other fields will be ignored.
- `param_db`: type of parameter database to construct when loading the policy.

This spec exists because of legacy issues. We use this spec to identify foreign policies in configuration, and turn it into `PolicyInfo` spec when pulling policy models.

### Trainer Worker

To configure a [trainer worker](../04_trainer_worker.md), the following fields must be specified:

- `policy_name`: same as in `PolicyWorker`, should match the policy of corresponding `PolicyWorker`.
- `trainer`: spec `Trainer` that defines an algorithm.
- `policy`: spec `Policy` that defines a policy, should match the policy of corresponding `PolicyWorker`.
- `sample_stream`: `SampleStream` spec, one and only one sample stream should be connected to one trainer worker.
- `foreign_policy`: `ForeignPolicy` spec, if not none, identify a foreign policy (not from this experiment) to load.
- `parameter_db`: `ParameterDB` spec.
- torch backend configuration:
    - `cudnn_benchmark`: related to CNN speed.
    - `cudnn_deterministic`: related to reproducibility.
- buffer related configuration:
    - `buffer_name`: string, identify type of buffer used in trainer.
    - `buffer_args`: arguments passed when constructing buffer.
- logging frequency:
    - `log_frequency_seconds`: length of time interval between logging.
    - `log_frequency_steps`: length of step interval between logging.
- parameter push frequency:
    - `push_frequency_seconds`:  length of time interval between pushing parameters.
    - `push_frequency_steps`: length of step interval between pushing parameters.
    - `push_tag_frequency_minutes`: length of time interval between pushing tagged version of policy models (exists permanently in storage). If none, never push tagged parameters.
- `worker_info`: `WorkerInformation` spec.
- `world_size`: DDP option. If world_size > 0, DDP will find exact world_size number of DDP peers, otherwise raise exception. If world_size = 0, DDP will find as much peers as possible. DDP peer did not connect to master will fail without terminating experiment. 

### Trainer

Spec for a trainer (an algorithm). See [User Guide: Policies and algorithms](04_policy_algorithm.md) for definition of a trainer (an algorithm). Fields:

- `type_`: type of the trainer
- `args`: argument that should be specified when constructing a trainer instance.

### Eval Mananger

User has to specify the following fields to configure an [evaluation manager](../06_eval_manager.md).

- `policy_name`: same as in `PolicyWorker`.
- `eval_sample_stream`: `SampleStream` spec that used to send evaluated samples.
- `eval_tag`
- `eval_target_tag`
- `eval_games_per_version`: number of games evaluated per version
- `eval_time_per_version_seconds`: number of seconds evaluated per version


### InferenceStream

Two types of inference streams are implemented in the local version of the system, see [System Components: Inference Stream](../07_inference_stream.md) for more details. To complete a `InferenceStream` spec, following fields should be specified:

- `type_`: only two types of inference streams are implemented: `InferenceStream.Type.LOCAL` and `InferenceStream.Type.INLINE`. If type is `LOCAL`, no other fields than `stream_name` are required to be specified in this spec. 
- `stream_name`: name of the stream.
- Inline inference information: For inline inference, a local policy worker is nested in it to complete CPU inference. For this reason, we need to specify informations for local policy worker:
    - `policy`
    - `policy_name`
    - `foreign_policy`
    - `policy_identifier`: If none, policy name will be sampled uniformly from available policies.
    - `accept_update_call`: whether local policy worker could be controlled by actor worker to update the policy model
    - `param_db`
    - `pull_interval_seconds`
    - `worker_info`

### SampleStream 

Two types of sample streams are implemented in the local version of the system, see [System Components: Sample Stream](../08_sample_stream.md) for more details. To complete a `SampleStream` spec, following fields should be specified:

- `type_`: type of sample stream, should be one of `SampleStream.Type.NULL` or `SampleStream.Type.LOCAL`
- `stream_name`: name of the stream.

## Examples
For basic examples of configuration files, we provide some benchmark experiments that you can play with in [legacy/experiments/](../../src/rlsrl/legacy/experiments). Since these experiments are already registered, you don't need to import the files with `--import_files` argument anymore. 

## Other Common Problems 

### Matching Agents to streams

`AgentSpec` is a configuration field of `ActorWorker`. It describes how each agent performs rollout and collect samples.

Firstly, note that the streams (inference/sample) come in lists in the configuration of actor workers. For example:

```python
from rlsrl.api.config import ActorWorker, AgentSpec

cfg = ActorWorker(inference_streams=["inf1", "inf2"],
                  sample_streams=["train", "eval"],
                  ...
                 )
```

So far, it is yet to decide which agents go to which streams. 

The example below matches all agents to inference 
stream `inf1` and sample stream `train`.
```python
                  ...
                  # Example 1
                  agent_specs=[AgentSpec(index_regex=f".*",
                                         inference_stream_idx=0,
                                         sample_stream_idx=0)],
                  ...
```
`index_regex` is a regular expression pattern, which will be matched with agent index(0, 1, ..., converted to string). `.*` will match everything. For more on regular expression, see [regular expression(python)](https://docs.python.org/3/library/re.html).

Note that the AgentSpec comes in a list. We can also match different agents to different Streams. The following example matches agents 0, 1, 2, 3 to `inf1` and `train`, and agents 4, 5, 6 to `inf2` and `eval`. 
```python
                  ...
                  # Example 2
                  agent_specs=[AgentSpec(index_regex="[0-3]",
                                         inference_stream_idx=0,
                                         sample_stream_idx=0),
                               AgentSpec(index_regex="[3-6]",
                                         inference_stream_idx=1,
                                         sample_stream_idx=1)
                               ],
                  ...
```
Although agent `3` is matched by both pattern, the latter won't overwrite the former. In other words, the system will follow the

### Version Control
If you want fixed parameter version for each trajectory sent to an SampleStream, follow the steps:
1. Set the InferenceStream type to Inline, specify `policy, policy_name, policy_identifier`
and set `pull_interval_seconds` to None.
2. Set the ring_size of the actor worker to be 1.

Read below for details.
- Parameter of inference stream other than inline inference (i.e. those connected to policy workers.) are controlled by Policy Workers. 
As policy workers usually serves many actor workers, possibly with ring_size > 1, they _cannot_ change the parameter
version based on the progress of any single environment. Instead, policy workers update parameter based on a frequency
sepcified in their configuration. As a result, the SampleBatch generated by a remote inference 
stream are highly likely to have varying parameter version.
- In some cases(e.g. evaluation, metadata updating, etc), a fixed sample version for each trajectory is beneficial. 
In this case, we should use Inline Inference Stream, which holds the neural network locally on cpu device. In addition,
We should set the `pull_interval_seconds` parameter of the inference stream to None, so that the `load_parameter`
function is only called by actor, upon reset. 
- Note that we can use a mixture of `inline` and `remote` inference streams in one Actor Worker. The parameter version
of the sample batch of different agents will follow different patterns. (stated above)
- If `ring_size > 1` OR `pull_interval_seconds is not None`, again, the samples generated by InlineInferenceStream will be varying. 

### Evaluation
Users may use evaluation managers to log evaluation results. To config your experiment to run evaluation, follow the steps:
1. Add ActorWorker with configuration ```is_evaluation=True, send_full_trajectory=True```
2. Add EvalManager and specify a tag in its configuration. e.g. ```tag="my_evaluation"```
3. Add PolicyWorkers with config ```policy_identifier="my_evaluation"```
4. Connect you Actor Worker with above-mentioned PolicyWorker and EvalManager with new inference/sample stream. It is suggested that you use "eval_" as prefix. _Note that if the name of one stream is the prefix of another, name-resolving streams won't be resolved correctly._
5. If you want a fixed number of evaluations for each version, set ```eval_games_per_version=100``` in EvalManager config.
6. Check the previous section on how to use a customized wandb_name.

### About tags
If you dive deeper into the parameter DB, the terms `tag` and `version` may get confusing. In general, `version` is 
dense and updated frequently as training proceeds, whereas `tags` are for workers to have some _consensus_ on how to use
the parameter versions.
These designs are still subject to changes and this is how things work currently:
1. A parameter DB is uniquely identified by ** experiment_name + trial_name + policy_name **
2. There is only one writer to each parameter DB: master of the trainer workers. The master trainer may `push without tag`
or `push with tag`. In both cases, the pushed parameter will be tagged as `latest`. And currently, this is considered as 
the `main policy version` that is being trained, and policy workers by default follows tag `latest`. 
3. `Push with tag` happens at a configurable frequency in trainer worker. See `push_frequency_seconds` and `push_frequency_steps` 
in TrainerWorker config. At this frequency, the trainer worker will attach a tag which is the system time.

- NOTE: Currently, latest is a tag that is maintained internally by the parameter DB. In other words, tag latest is 
automatically added to each push, and it is not considered a push with tag if no additional tags are specified.

4. If the master trainer pushes `without tag`, the pushed parameter will be garbage-collected when it is outdate. 
In other words, the parameter will be considered as "saved for future use" only if the master trainer `push with tag`.
5. In the case of parameter DB that supports metadata, the pushed parameter will enter the metadata-database _only if_ it is pushed with tags. (metadata database is not implemented in current local version) 
6. If a version is pushed `without tag` and tagged later (e.g. for evaluation purpose), its metadata cannot be tracked, and its parameter will be kept in the database until its last tag is removed. During its life time, it can be retrieved by its tag and/or version, but not though metadata query. (metadata database is not implemented in current local version) 
7. In our current implementation, version is the number of backward pass to get the parameter, which is consistent with the `policy_verison` attributed of a policy, `policy_version` of the RolloutResult, and the `policy_version` of a SampleBatch.

# Related References
- [User Guide: Logging](02_logging.md)
- [User Guide: Environments](03_environments.md)
- [User Guide: Policies and algorithms](04_policy_algorithm.md)
- [System Components: Overview](../00_system_overview.md)

# Related Files and Directories
- [api/config.py](../../src/rlsrl/api/config.py)
- [legacy/experiments/](../../src/rlsrl/legacy/experiments)

# What's Next
- [User Guide: Logging](02_logging.md)