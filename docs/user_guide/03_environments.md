# Environment

Our **Environment** API provides an interface for user to implement any RL environments, including multi-agent environments. Our environment API as the following features:

1. Agents in an environment can have different observation shapes, action spaces, etc.;
2. Environment can be asynchronous. In some multi-agent environments, e.g. Hanabi, only part of the agents make action at each step. We solve these cases by supporting `None` return values for agents, and does not generate action for such agents in an environment step. 

and the following limitations:

1. Environments have to be homogeneous within an execution, the number of agents in an environment cannot change during an execution;
2. Observation space of each agent cannot change (There is no dynamic inference stream matching.) 

## Environment Dataclasses

In [api/env_utils.py](../../src/rlsrl/api/env_utils.py), we provide utility data classes to represent actions and action spaces in the form of data structures in our system. We support both discrete (class `DiscreteAction` and `DiscreteActionSpace`) and continuous (class `ContinuousAction` and `ContinuousActionSpace`).

Moreover, environments pass the result of each reset and step for an single agent to the system as a dataclass `StepResult`, which includes 4 fields: 

- `obs`: one step observation.
- `reward`: one step reward.
- `done`: whether this agent is done in this episode. 
- `info`: other informations.   

## Implementing Environments

The procedure of implementing new environments is similar to writing a new config file. A new **Environment** is defined by a subclass of `Environment` from [api/environment.py](../../src/rlsrl/api/environment.py). To run experiments with a new environment, first register it with method `rlsrl.api.environment.register(name, env_class)`, then specify it in experiment config file with `Environment` spec. For example, a new environment `SomeEnvironment(Environment)` is implemented in file `some_env.py`: 

```python
# some_env.py

from rlsrl.api.environment import *

class SomeEnvironment(Environment):
    def __init__(self, **kwargs):
        ...

    def step(self, actions):
        ...
        return StepResult(obs=..., reward=..., ...)
    
    def reset(self):
        ...
        return StepResult(obs=..., reward=..., ...)
```

You may implement other methods in class `Environment` as well, see [api/environment.py](../../src/rlsrl/api/environment.py) for details. In the end of `some_env.py`, environment needs to be registered:

```python
# some_env.py

register("some_env", SomeEnvironment)
```

After that, you can write your experiment config file, and write `ActorWorker` spec with your environment spec:

```python
# some_env_config.py

class SomeEnvExperiment(Experiment):
    def initial_setup(self):
        ...
        return ExperimentConfig(actor_workers=
                                    [ActorWorker(env=Environment(
                                                            type_="some_env",
                                                            args=args,
                                                            )
                                                ...
                                                )
                                    ...],
                                ...
                               )

register("some_env_expr", SomeEnvExperiment)

```

When running experiment with command line, `some_env.py` should be included in the `--import_files` option as well: 

`srl-local run -e some_env_expr -f hello --import_files some_env.py;some_env_config.py`

Of course, you could refer to environments that has already been implemented in the system in directory [legacy/environment/](../../src/rlsrl/legacy/environment):

- **[Gym Atari](../../src/rlsrl/legacy/environment/atari/atari_env.py)**
- **[Google football](../../src/rlsrl/legacy/environment/google_football/gfootball_env.py)**
- **[Gym MuJoCo](../../src/rlsrl/legacy/gym_mujoco/gym_mujoco_env.py)**
- **[Hide and Seek](../../src/rlsrl/legacy/environment/hide_and_seek/hns_env.py)**
- **[SMAC](../../src/rlsrl/legacy/environment/smac/smac_env.py)**


# Related References
- [System Components: Actor Worker](../02_actor_worker.md)

# Related Files and Directories
- [api/env_utils.py](../../src/rlsrl/api/env_utils.py)
- [api/environment.py](../../src/rlsrl/api/environment.py)
- [legacy/environment/](../../src/rlsrl/legacy/environment/)

# What's next
- [User Guide: Policy and algorithm development](04_policy_algorithm.md)
