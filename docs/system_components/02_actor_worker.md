# System Components: Actor Worker

_"Reality is merely a simulation of God who himself is the simulation of humanity emulated through a brief history of time."_

An **Actor Worker** simulates multiple environments to produce samples (training data) for RL training. In a single actor workers, multiple copies of environments receives actions from [Policy Workers](03_policy_worker.md) via [Inference Clients](07_inference_stream.md), simulates to obtain observations from the environments and send them to [Policy Workers](03_policy_worker.md) via [Inference Clients](07_inference_stream.md). After collecting enough simulation data (action, observations, etc.), actor workers batch them into a sample and send them to [Trainer Workers](04_trainer_workers) via [Sample Producers](08_sample_stream.md) as training data.

## Agent (class `Agent`)

An **Agent** is a minimal acting unit in an simulated environment. Each agent has its own policy, which means it should communicate with [Inference Stream](07_inference_stream.md) and [Sample Stream](08_sample_stream.md) corresponding to the policy. An agent has two major tasks:
1. Get inference results from its Inference stream, corresponding to `Agent.get_action()`. After retriving one step inference result (an action), put the result into last observation in the memory, which is the request corresponding to result, to form a complete [Sample Batch](../user_guide/09_basic_apis.md). Pass the action to Environment Target and expect for next step result. Finally, update policy state from inference stream.
2. Process a new step result from its Environment Target. One new step result contains new observation and reward of last step. Upon receiving a new observation, an agent first check whether it should post sample batches in its memory to Sample Stream. The conditions of sending sample batches are related to agent configuration (See [AgentSpec configuration](../user_guide/config_your_experiment.md#AgentSpec)). 

### Initialization

Refer to the initialization docstring of [AgentSpec configuration](../user_guide/config_your_experiment.md#AgentSpec) on how to config an Agent.

## Environment Target (class `_EnvTarget`)

An **Environment Target** hold a single **Environment** instance, and manages all Agent instances in an environment. An Environment Target Exposes 5 methods for the actor worker to call: 
1. `all_done()`: Check if all agents in this environment are done.
2. `unmask_all_info()`: Unmask the most recent episode info of all agents.
3. `reset()`: Reset the environment.
4. `step()`: Get actions from agents, and perform one environment step.
5. `ready_to_step()`: Check if all agents are ready to perform a step in the environment.

### Initialization

The initialization of an Environment Target requires an Environment instance, a maximum number of steps that the environment is allowed to run per episode (to avoid a dead loop in an environment), and a list of Agents in this Environment Target.

## Environment Ring (class `_EnvRing`)

In a trivial implementation of actor workers, simulation and policy inference are executed sychronously. After completing simulation for one step, actor worker sends inference requests to its corresponding Inference Streams, waits for replys and then continue to simulate the next step. In this procedure, resources occupied by actor worker will idle when waiting for replys. To utilize resource as much as possible, the actor worker could run simulation in an asynchronous fashion, which means running simulation on other environments while waiting for inference replys. **Environment Ring** is a data structure that supports asynchronous simulation.

One Environment Ring contains multiple copies of identical Environment Targets_(i.e. environment simulators). Each _Environment Ring_ has a pointer that points to one target. After one simulation step, the ring rotates and the pointer points to the next target.

### Initialization

Specify list of Environment Targets in the ring to initialize Environment Ring.

## class `Actor Worker`

This class is inherited from `worker_base.Worker` ([Worker Base](01_worker_base.md)).

In each `_poll()`, the actor worker execute following procedure for `_MAX_POLL_STEPS` (default = 16) times:
1. Check if all Agents in this Environment Target is done. 
    1. If yes, unmask all info and reset this Environment Target. If using inline inference, load new parameters for local policy worker. 
    2. If not, check if any agents in this Environment Target is waiting for inference reply. 
        1. If yes, break to wait for agents to be ready.
        2. If no, perform one target step.
2. Flush inference clients every determined number of steps. (See [Inference Stream](07_inference_stream.md))
3. Rotate Environment Ring.

### Initialization

Refer to the initialization docstring of [ActorWorker configuration](../user_guide/config_your_experiment.md#ActorWorker) on how to config an _Actor Worker_.

Initialization process is (`_configure()`):
1. Make Inference Clients and Sample Producers. The actor worker keeps a reference to these streams.f
2. Make agents as specified by AgentSpec. Each agent is matched with an Inference Client and a Sample Producer.
3. Create many Environment Targets with the specified Environment and agents created in 2.
4. Create the Environment Ring.

# Related References

- [System Overview](00_system_overview.md)
- [Worker Base](01_worker_base.md)
- [Policy Worker](03_policy_worker.md)
- [Trainer Worker](04_trainer_worker.md)
- [Inference Stream](07_inference_stream.md)
- [Sample Stream](08_sample_stream.md)

# Related File and Directory
- [system/basic/impl/actor_worker.py](../../src/rlsrl/system/impl/actor_worker.py)

# What's Next

- [Policy Worker](03_policy_worker.md)
