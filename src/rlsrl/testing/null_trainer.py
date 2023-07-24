import torch

from rlsrl.api.trainer import Trainer, TrainerStepResult
from rlsrl.api.trainer import register
from rlsrl.api.policy import make as make_policy
import rlsrl.api.policy as policy_api


class NullPolicy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def set_state_dict(self, param):
        self.load_state_dict(param)


class NullTrainer(Trainer):

    @property
    def policy(self) -> policy_api.Policy:
        return self._policy

    def __init__(self, policy, **kwargs):
        self._policy = policy
        self.steps = 0

    def step(self, sample):
        self.steps += 1
        return TrainerStepResult(stats={}, step=0)

    def distributed(self, **kwargs):
        pass

    def get_checkpoint(self, *args, **kwargs):
        return {}

    def load_checkpoint(self, *args, **kwargs):
        pass


register('null_trainer', NullTrainer)
