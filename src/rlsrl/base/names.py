# This file standardizes the name-resolve names used by different components of the system.
import getpass

USER_NAMESPACE = getpass.getuser()


def registry_root(user):
    return f"trial_registry/{user}"


def trial_registry(experiment_name, trial_name):
    return f"trial_registry/{USER_NAMESPACE}/{experiment_name}/{trial_name}"


def trial_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}"


def worker_status(experiment_name, trial_name, worker_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/status/{worker_name}"


def worker_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker/"


def worker(experiment_name, trial_name, worker_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker/{worker_name}"


def worker2(experiment_name, trial_name, worker_type, worker_index):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker/{worker_type}/{worker_index}"


def inference_stream(experiment_name, trial_name, stream_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/inference_stream/{stream_name}"


def inference_stream_constant(experiment_name, trial_name, stream_name,
                              constant_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/inference_stream_consts/{stream_name}/{constant_name}"


def sample_stream(experiment_name, trial_name, stream_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/sample_stream/{stream_name}"


def trainer_ddp_peer(experiment_name, trial_name, policy_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/trainer_ddp_peer/{policy_name}"


def trainer_ddp_master(experiment_name, trial_name, policy_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/trainer_ddp_master/{policy_name}"


def curriculum_stage(experiment_name, trial_name, curriculum_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/curriculum/{curriculum_name}"


def worker_key(experiment_name, trial_name, key):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker_key/{key}"


def parameter_subscription(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/parameter_sub"


def parameter_server(experiment_name, trial_name, parameter_id_str):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/parameter_server/{parameter_id_str}"


def shared_memory(experiment_name, trial_name, dock_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/shared_memory/{dock_name}"


def pinned_shm_qsize(experiment_name, trial_name, stream_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/pinned_shm_qsize/{stream_name}"
