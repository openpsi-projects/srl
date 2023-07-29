# SRL (**R**ea**L**ly **S**calable **RL**): Scaling Distributed Reinforcement Learning to Over Ten Thousand Cores

**SRL** is an _efficient_, _scalable_ and _extensible_ distributed **Reinforcement Learning** system. **SRL** supports running several state-of-the-art RL algorithms on some common environments with one simple configuration file, and also exposes general APIs for users to develop their self-defined environments, policies and algorithms. **SRL** even allows users to implement new system components to support their algorithm designs, if current system architecture is not sufficient. 

Currently, our scheduler with `slurm` is not released. We are planning to implement a `ray` version launcher for users to easily deploy SRL on a large scale!   

## Algorithms and Environments

In this repository, one algorithm (**[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)**) and five environments (**[Gym Atari](https://www.gymlibrary.dev/environments/atari/), [Google football](https://github.com/google-research/football), [Gym MuJoCo](https://www.gymlibrary.dev/environments/mujoco/), [Hide and Seek](https://openai.com/blog/emergent-tool-use/), [SMAC](https://github.com/oxwhirl/smac)**) are implemented as examples. In the future, more environment and algorithm supports will be added to build an RL library with SRL.

## Installation

Before installation, make sure you have `python>=3.8` and `torch>=1.10.0, gym` installed. [Wandb](https://wandb.ai/) is also supported, please install `wandb` package if you intend to use it for logging. You should also install environments you intend to run. For more information, check links about supported envrionment in previous section. (Note that **Google football** environment requires a older version of `gym==0.21.0`)

Contents in this repository could be installed as a python package. To install, you should clone this repository and install the package by:

`git clone https://github.com/openpsi-projects/srl.git`

`cd srl && pip install -e .`

## Running an Experiment

After installing **SRL** and atari environment, to run a simple experiment we provide as an example: 

`srl-local run -e atari-mini -f test`

This command line will start a run of simple PPO training on environment atari, defined by:

- Experiment config: [src/rlsrl/legacy/experiments/atari_benchmark.py](src/rlsrl/legacy/experiments/atari_benchmark.py)

- Atari environment implementation: [src/rlsrl/legacy/environment/atari/atari_env.py](src/rlsrl/legacy/environment/atari/atari_env.py)

- Algorithm and policy implementation: [src/rlsrl/legacy/algorithm/ppo/](src/rlsrl/legacy/algorithm/ppo/)

## Documentation

For more user guides:
- [Users Guide](docs/user_guide/00_overview.md)

For more information about **SRL**:
- [System Components](docs/system_components/00_system_overview.md)

## Full paper

Full paper: **SRL: Scaling Distributed Reinforcement Learning to Over Ten thousand cores** available in arxiv! Link: **[https://arxiv.org/abs/2306.16688](https://arxiv.org/abs/2306.16688)**
