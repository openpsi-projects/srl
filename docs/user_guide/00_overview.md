# User Guide: Overview

This user guide is a detailed manual for both new-comers that wish to train an RL agent with well-implemented algorithms in supported envrionments, and advanced users that aim to design and implement their own algorithms and even system components. It includes documentation of APIs exposed by the system and instructions of common system usage.

## Command Line Options

To run an experiment in **SRL** after installation:

`srl-local run -e <experiment name> -f <trial name>`

Options include:

- `-e (--experiment_name)`: Experiment name, explained in [User Guide: Config Your Experiment](01_experiment_config.md).
- `-f (--trial_name)`: Experiment name, explained in [User Guide: Config Your Experiment](01_experiment_config.md)
- `--wandb_mode`: Wandb Logging mode, includes online, offline and disabled (default). Explained in [User Guide: Logging](02_logging.md).
- `--import_files`: Extra files to import, include files that defines experiments, environments, policies and algorithms. 

## For Beginners

If you are a first-time user, its highly recommanded to try running our system with existing configuration files to observe logging and behaviors of system components. If you are interested, we also provide [documentation about our system components](../system_components/00_system_overview.md). 

After getting familiar with our system, you might want to try more different parameters and training options to get better results in your experiments. Read [User Guide: Config Your Experiment](01_experiment_config.md) to learn how to run an experiment and write your own configuration files. We support logging in terminal output as well as using visualize tools [wandb](https://wandb.ai/site). Read [User Guide: Logging](02_logging.md) to understand how to configure logging and learn system and training data that could be logged in our system.

Experiment Configuration examples: [legacy/experiments/](../../src/rlsrl/legacy/experiments/).

## For Advanced Users

For advanced users, if you want to run experiments on environments that are not inherently supported in our system, we provide environment APIs, which is a wrapper of [gym API](url=https://github.com/openai/gym#api). Read [User Guide: Environments](03_environments.md) to learn to use our environment APIs. Also, you might want to re-implement novel RL algorithms in our systems, or design and implement your own new algorithms. To learn how to implement new policies and algorithms, please read [User Guide: Policy and algorithm development](04_policy_algorithm.md). 

In our system, we use `NamedArray` as a basic data structure, which is an extension to `numpy` arrays. `NamedArray` is almost used everywhere in our system, and learning how it works will greatly help you get your hands on coding. Read [User Guide: NamedArray](05_named_array.md) for detailed information about `NamedArray`.  

Environment implementation examples: [legacy/environments/](../../src/rlsrl/legacy/environments/), Policy and Algorithm implementation examples: [legacy/algorithm/](../../src/rlsrl/legacy/algorithm/).

## For System-level Users

Sometimes you might find our system's architecture (actor, policy, trainer workers with a simple parameter server) cannot meet the needs of your new algorithm. This usually happens when your algorithms require additional data processing or transformation, and have computation workloads other than simulation, policy inference and training. For example, Muzero (Reference: [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://www.nature.com/articles/s41586-020-03051-4)) requires reanalyzation before sending samples to trainers. This is where you should implement your own workers and plug them into our system. We expose APIs for workers and communication APIs in sample/inference streams to ensure that you could implement any worker with any communication patterns you desire. For more detailed information and references, please read [documentation about our system components](../system_components/00_system_overview.md) thoroughly. 

# Related References
- [User Guide: Experiment Config](01_experiment_config.md)
- [User Guide: Logging](02_logging.md)
- [User Guide: Environments](03_environments.md)
- [User Guide: Policy and algorithm development](04_policy_algorithm.md)
- [User Guide: NamedArray](05_named_array.md)

# What's Next
- [User Guide: Experiment Config](01_experiment_config.md)
