import collections
import dataclasses
import importlib
import logging
import os
import traceback


@dataclasses.dataclass
class WorkerSpec:
    """Description of a worker implementation.
    """
    short_name: str  # short name is used in file names.
    config_field_name: str  # Used in experiment/scheduling configuration(api.config).
    class_name: str  # The class name of the implementation.
    module: str  # The module path to find the worker class.

    def load_worker(self):
        module = importlib.import_module(self.module)
        return getattr(module, self.class_name)


actor_worker = WorkerSpec(short_name='aw',
                          class_name="ActorWorker",
                          config_field_name="actors",
                          module="rlsrl.system.impl.actor_worker")
# buffer_worker = WorkerSpec(short_name='bw',
#                            class_name="BufferWorker",
#                            config_field_name="buffers",
#                            module="rlsrl.system.impl.buffer_worker")
eval_manager = WorkerSpec(short_name='em',
                          class_name="EvalManager",
                          config_field_name="eval_managers",
                          module="rlsrl.system.impl.eval_manager")
policy_worker = WorkerSpec(short_name='pw',
                           class_name="PolicyWorker",
                           config_field_name="policies",
                           module="rlsrl.system.impl.policy_worker")
trainer_worker = WorkerSpec(short_name='tw',
                            class_name="TrainerWorker",
                            config_field_name="trainers",
                            module="rlsrl.system.impl.trainer_worker")
# population_manager = WorkerSpec(short_name="pm",
#                                 class_name="PopulationManager",
#                                 config_field_name="population_manager",
#                                 module="rlsrl.system.impl.population_manager")

RL_WORKERS = collections.OrderedDict()
RL_WORKERS["trainer"] = trainer_worker
# RL_WORKERS["buffer"] = buffer_worker
RL_WORKERS["policy"] = policy_worker
RL_WORKERS["eval_manager"] = eval_manager
# RL_WORKERS["population_manager"] = population_manager
RL_WORKERS["actor"] = actor_worker


def run_worker(worker_type,
               experiment_name,
               trial_name,
               worker_name,
               ctrl,
               env_vars=None):
    """Run one worker
    Args:
        worker_type: string, one of the worker types listed above,
        experiment_name: string, the experiment this worker belongs to,
        trial_name: string, the specific trial this worker belongs to,
        worker_name: name given to the worker, typically "<worker_type>/<worker_index>"
    """
    import torch
    torch.cuda.init()
    if env_vars is not None:
        for k, v in env_vars.items():
            os.environ[k] = v

    if worker_type == "master":
        from rlsrl.system.impl.master_worker import MasterWorker as worker_class
    else:
        worker_class = RL_WORKERS[worker_type].load_worker()
    worker = worker_class(ctrl=ctrl)
    try:
        worker.run()
    except Exception as e:
        logging.error("Worker %s failed with exception: %s", worker_name, e)
        logging.error(traceback.format_exc())
        raise e
