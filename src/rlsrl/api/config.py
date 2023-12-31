from typing import Dict, List, Optional, Any, Union
import copy
import dataclasses
import enum
import math
import sys
import yaml


@dataclasses.dataclass
class Condition:

    class Type(enum.Enum):
        SimpleBound = 1  # A simple condition that compares the value to a upper and lower limit.
        Converged = 2  # Condition that checks if the value is converged.

    type_: Type
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Curriculum:

    class Type(enum.Enum):
        Linear = 1

    type_: Type
    name: str
    stages: Union[List[str], str] = "training"
    conditions: List[Condition] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Environment:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DataAugmenter:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Policy:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    init_ckpt_dir: Optional[str] = None


@dataclasses.dataclass
class Trainer:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrajPostprocessor:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class MetaSolver:

    class Type(enum.Enum):
        UNIFORM = 1
        NASH = 2

    type_: Type
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PopulationAlgorithm:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ParameterDB:

    class Type(enum.Enum):
        FILESYSTEM = 1  # Saves parameters to shared filesystem.

        LOCAL_TESTING = 99

    type_: Type


@dataclasses.dataclass
class ParameterServer:

    class Type(enum.Enum):
        MultiCast = 1

    type_: Type = Type.MultiCast
    backend_db: ParameterDB = ParameterDB(type_=ParameterDB.Type.FILESYSTEM)


@dataclasses.dataclass
class ParameterServiceClient:

    class Type(enum.Enum):
        MultiCast = 1

    type_: Type = Type.MultiCast


@dataclasses.dataclass
class ForeignPolicy:
    """A Policy is foreign if any of the configurations below differs from the worker's original configuration.
    Workers behave differently when receiving a foreign parameter_db.
    1. Trainer will read this policy when initializing, except when it is resuming a previous trial. Trained Parameters
    will be pushed to its domestic parameter db.
    2. Policy Worker/ InlineInference: foreign policy will overwrite domestic policy. i.e. worker will always load
    foreign policy.

    NOTE:
        -1. If absolute_path is not None, All others configurations will be ignored.
        0. When absent(default to None), the absent fields will be replaced by domestic values.
        1. Foreign policies are taken as `static`. No workers should try to update a foreign policy. And foreign policy
        name should not appear in the domestic experiment.
        2. Currently only Trainer Worker has foreign policy implemented. If you need to use foreign policy for
        inference only, a workaround is to use a dummy trainer, with no gpu assigned. to transfer the policy to
        domestic parameter db.
    """
    foreign_experiment_name: Optional[str] = None
    foreign_trial_name: Optional[str] = None
    foreign_policy_name: Optional[str] = None
    foreign_policy_identifier: Optional[str] = None
    absolute_path: Optional[str] = None

    param_db: ParameterDB = ParameterDB(type_=ParameterDB.Type.FILESYSTEM)


@dataclasses.dataclass
class WorkerInformation:
    """The basic information of an worker. To improve config readability, the experiment starter will fill the
    fields, instead of letting the users do so in experiment configs.
    """
    experiment_name: str = ""
    trial_name: str = ""  # Name of the trial of the experiment; e.g. "{USER}-0".
    worker_type: str = ""  # E.g. "policy", "actor", or "trainer".
    worker_index: int = -1  # The index of the worker of the specific type, starting from 0.
    worker_count: int = 0  # Total number of workers; hence, 0 <= worker_index < worker_count.
    worker_tag: Optional[
        str] = None  # For actor and policy worker, can be "training" or "evaluation".
    policy_name: Optional[
        str] = None  # For trainer and policy worker, the name of the policy.
    wandb_entity: Optional[
        str] = None  # wandb_{config} are optional. They overwrite system wandb_configuration.
    wandb_project: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    log_wandb: Optional[bool] = None
    log_terminal: Optional[bool] = None
    device: str = 'cpu'
    node_rank: int = 0
    name_resolve_address: Optional[str] = None
    name_resolve_port: Optional[int] = None

    def system_setup(self, experiment_name, trial_name, worker_type,
                     worker_index, worker_count, policy_name):
        """Setup system related worker information, while leaving the rest untouched.
        """
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.worker_type = worker_type
        self.worker_index = worker_index
        self.worker_count = worker_count
        self.policy_name = policy_name

    def set_name_resolve_address(self, address, port):
        # print(f"Setting {self.worker_type}: {self.worker_index} name resolve address to {address}:{port}")
        self.name_resolve_address = address
        self.name_resolve_port = port


@dataclasses.dataclass
class IpStreamPlugin:
    address: str


@dataclasses.dataclass
class InlineInferenceStreamPlugin:
    policy: Policy
    policy_name: str
    foreign_policy: Optional[ForeignPolicy] = None
    accept_update_call: bool = True
    # If None, policy name will be sampled uniformly from available policies.
    policy_identifier: Union[str, Dict, List] = "latest"
    param_db: ParameterDB = ParameterDB(ParameterDB.Type.FILESYSTEM)
    pull_interval_seconds: Optional[int] = None
    population: Optional[List[str]] = None
    policy_sample_probs: Optional[List[float]] = None
    parameter_service_client: Optional[ParameterServiceClient] = None


@dataclasses.dataclass
class InferenceStream:

    class Type(enum.Enum):
        IP = 1  # An stream specifying the exact location of the server.
        NAME = 2  # An stream where clients resolve the server locations via name-resolve.
        INLINE = 3
        SHARED_MEMORY = 4

    type_: Type
    stream_name: str = ""
    stream_index: Optional[int] = None
    serialization_method: str = "pickle"
    worker_info: Optional[WorkerInformation] = None

    plugin: Optional[Union[IpStreamPlugin, InlineInferenceStreamPlugin]] = None

    def __post_init__(self):
        if self.type_ == InferenceStream.Type.IP:
            assert isinstance(self.plugin, IpStreamPlugin)
        elif self.type_ == InferenceStream.Type.INLINE:
            assert isinstance(self.plugin, InlineInferenceStreamPlugin)
        else:
            assert self.plugin is None
        if self.plugin is not None:
            for field in dataclasses.fields(self.plugin):
                setattr(self, field.name, getattr(self.plugin, field.name))


@dataclasses.dataclass
class SharedMemorySampleStreamPlugin:
    qsize: int
    reuses: int = 1
    batch_size: Optional[int] = None


@dataclasses.dataclass
class _InlineSampleStreamPlugin:
    # Use only for testing.
    trainer: Trainer
    policy: Policy


@dataclasses.dataclass
class SampleStream:

    class Type(enum.Enum):
        NULL = 0  # Only producer side is implemented. NULL Producer discards all samples.
        IP = 1  # An stream specifying the exact location of the server.
        NAME = 2  # An stream where clients resolve the server locations via name-resolve.
        NAME_ROUND_ROBIN = 4  # Send sample to consumers in Round-Robin manner. Only the producers side is implemented.
        NAME_MULTI_AGENT = 5  # Producer who batch all agents before sending.
        NAME_BROADCAST = 6  # Producer who broadcasts sample to all consumers. Only the producers side is implemented.
        SHARED_MEMORY = 7

        INLINE_TESTING = 99  # Runs the training inline, for testing only.

    type_: Type
    stream_name: str = ""
    stream_index: Optional[int] = None
    serialization_method: str = "pickle"  # Check `base.namedarray` for available methods.
    worker_info: Optional[WorkerInformation] = None

    plugin: Optional[Union[IpStreamPlugin, SharedMemorySampleStreamPlugin,
                           _InlineSampleStreamPlugin]] = None

    def __post_init__(self):
        if self.type_ == SampleStream.Type.IP:
            assert isinstance(self.plugin, IpStreamPlugin)
        elif self.type_ == SampleStream.Type.SHARED_MEMORY:
            assert isinstance(self.plugin, SharedMemorySampleStreamPlugin)
        elif self.type_ == SampleStream.Type.INLINE_TESTING:
            assert isinstance(self.plugin, _InlineSampleStreamPlugin)
        else:
            assert self.plugin is None
        if self.plugin is not None:
            for field in dataclasses.fields(self.plugin):
                setattr(self, field.name, getattr(self.plugin, field.name))


@dataclasses.dataclass
class AgentSpec:
    """The configuration of agents of each actor worker consists of a list of AgentSpec-s. Each AgentSpec
    matches some of the agents, as well as provides inference stream and sample stream configs.
    """
    index_regex: str
    inference_stream_idx: int  # Multiple inference stream is not ready yet.
    sample_stream_idx: Union[int, List[int]]
    sample_steps: int = 200
    bootstrap_steps: int = 1
    burn_in_steps: int = 0
    deterministic_action: bool = False
    send_after_done: bool = False
    send_full_trajectory: bool = False
    pad_trajectory: bool = False  # only used if send_full_trajectory
    trajectory_postprocessor: Union[str, TrajPostprocessor] = 'null'
    compute_gae_before_send: bool = False
    gae_args: Optional[Dict] = dataclasses.field(default_factory=dict)
    send_concise_info: bool = False

    def __post_init__(self):
        if not self.send_after_done and not self.send_full_trajectory and self.trajectory_postprocessor != "null":
            raise ValueError(
                "Either `send_after_done` or `send_full_trajectory` should be True if "
                "trajectory postprocessor is activated!")


@dataclasses.dataclass
class ActorWorker:
    """Provides the full configuration for an actor worker.
    """
    env: Union[
        str, Environment,
        List[Environment]]  # If str, equivalent to Environment(type_=str),
    # If List[Environment], len(env) should be equal to ring size.
    # `make_env` method will use env[i] as config for i-th environment
    # in env ring.
    sample_streams: List[Union[str, SampleStream]]
    inference_streams: List[Union[str, InferenceStream]]
    agent_specs: List[AgentSpec]
    max_num_steps: int = 100000
    ring_size: Optional[int] = 2
    execution_method: str = "ring"  # Can be "ring" or "threaded"
    # Actor worker will split the ring into `inference_splits` parts # and flush the inference clients for each part.
    inference_splits: int = 2
    decorrelate_seconds: float = 0.0
    curriculum_config: Optional[Curriculum] = None
    worker_info: Optional[
        WorkerInformation] = None  # Specify your wandb_config here, or leave None.

    def __post_init__(self):
        if isinstance(self.env, List):
            self.ring_size = len(self.env)
        else:
            self.env = [copy.deepcopy(self.env) for _ in range(self.ring_size)]


@dataclasses.dataclass
class BufferWorker:
    """Provides the full configuration for a buffer worker.
    """
    from_sample_stream: Union[str, SampleStream]
    to_sample_stream: Union[str, SampleStream]
    buffer_name: str = "priority_queue"
    buffer_args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    parameter_db: Optional[ParameterDB] = ParameterDB(
        ParameterDB.Type.FILESYSTEM)
    data_augmenter: Optional[DataAugmenter] = None
    policy: Optional[Policy] = None
    policy_name: Optional[str] = None
    policy_identifier: Optional[str] = None
    pull_frequency_seconds: float = 5
    reanalyze_target: str = "muzero"
    unpack_batch_before_post: bool = True
    parameter_service_client: Optional[ParameterServiceClient] = None
    worker_info: Optional[
        WorkerInformation] = None  # Specify your wandb_config here, or leave None.


@dataclasses.dataclass
class PolicyWorker:
    """Provides the full configuration for a policy worker.
    """
    policy_name: str
    inference_stream: Union[str, InferenceStream]
    policy: Union[str, Policy]  # If str, equivalent to Policy(type_=str).
    parameter_db: ParameterDB = ParameterDB(ParameterDB.Type.FILESYSTEM)
    batch_size: int = 10240  # Batch size is an upper bound. Set larger unless you experience OOM issue.
    max_inference_delay: float = 0.1
    policy_identifier: Union[str, Dict, List] = "latest"
    pull_frequency_seconds: float = 1
    pull_max_failures: int = 100
    foreign_policy: Optional[ForeignPolicy] = None
    parameter_service_client: Optional[ParameterServiceClient] = None
    worker_info: Optional[
        WorkerInformation] = None  # Specify your wandb_config here, or leave None.


@dataclasses.dataclass
class TrainerWorker:
    """Provides the full configuration for a trainer worker.
    """
    policy_name: str
    trainer: Union[str, Trainer]  # If str, equivalent to Trainer(type_=str).
    policy: Union[str, Policy]  # If str, equivalent to Policy(type_=str).
    sample_stream: Union[str, SampleStream]
    foreign_policy: Optional[ForeignPolicy] = None
    parameter_db: ParameterDB = ParameterDB(ParameterDB.Type.FILESYSTEM)
    # cudnn benchmark is related to the speed of CNN
    cudnn_benchmark: bool = True
    # cudnn deterministic is related to reproducibility
    cudnn_determinisitc: bool = True
    buffer_name: str = "priority_queue"
    buffer_args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    save_buffer_on_exit: bool = False
    load_buffer_on_restart: bool = False
    buffer_zero_copy: bool = False
    log_frequency_seconds: int = 5
    log_frequency_steps: int = None
    push_frequency_seconds: Optional[float] = 1.
    push_frequency_steps: Optional[int] = 1
    push_tag_frequency_minutes: Optional[
        int] = None  # Adds a tagged policy version regularly.
    preemption_steps: float = math.inf
    worker_info: Optional[
        WorkerInformation] = None  # Specify your wandb_config here, or leave None.

    def __post_init__(self):
        if self.buffer_zero_copy:
            assert self.sample_stream.type_ == SampleStream.Type.SHARED_MEMORY
            # Align buffer and sample stream configuration to store data in a zero-copy manner.
            self.sample_stream.reuses = self.buffer_args.get(
                "reuses", self.sample_stream.reuses)
            self.sample_stream.batch_size = self.buffer_args.get(
                "batch_size", self.sample_stream.batch_size)
            if self.sample_stream.batch_size is None:
                self.sample_stream.batch_size = 1
            self.sample_stream.qsize = self.buffer_args.get(
                "max_size", self.sample_stream.qsize /
                self.sample_stream.batch_size) * self.sample_stream.batch_size
            self.buffer_name = "simple_queue"
            self.buffer_args = dict(max_size=100)
        elif not isinstance(
                self.sample_stream, str
        ) and self.sample_stream.type_ == SampleStream.Type.SHARED_MEMORY:
            assert self.sample_stream.reuses == 1
            assert self.sample_stream.batch_size is None


@dataclasses.dataclass
class MasterWorker:
    address: str = "localhost"
    port: int = 51234
    parameter_server: ParameterServer = ParameterServer()
    worker_info: Optional[WorkerInformation] = WorkerInformation()


@dataclasses.dataclass
class DummyWorker:
    worker_info: Optional[WorkerInformation] = WorkerInformation()


@dataclasses.dataclass
class EvaluationManager:
    policy_name: str
    eval_sample_stream: Union[str, SampleStream]
    parameter_db: ParameterDB = ParameterDB(ParameterDB.Type.FILESYSTEM)
    eval_target_tag: str = "latest"
    eval_tag: str = "eval"
    eval_games_per_version: Optional[int] = 100
    eval_time_per_version_seconds: Optional[float] = None
    unique_policy_version: Optional[bool] = True
    curriculum_config: Optional[Curriculum] = None
    log_evaluation: bool = True
    update_metadata: bool = True
    worker_info: Optional[
        WorkerInformation] = None  # Specify your wandb_config here, or leave None.


@dataclasses.dataclass
class PopulationManager:
    population: List[str]
    population_algorithm: Union[str, PopulationAlgorithm]
    population_sample_stream: SampleStream
    actors: List[ActorWorker]
    policies: List[PolicyWorker]
    trainers: List[TrainerWorker]
    eval_managers: Optional[List[EvaluationManager]] = dataclasses.field(
        default_factory=list)
    worker_info: Optional[
        WorkerInformation] = None  # Specify your wandb_config here, or leave None.


@dataclasses.dataclass
class ParameterServerWorker:
    parameter_server: ParameterServer = dataclasses.field(
        default_factory=ParameterServer)
    worker_info: Optional[
        WorkerInformation] = None  # Specify your wandb_config here, or leave None.


@dataclasses.dataclass
class ExperimentConfig:
    actors: List[ActorWorker] = dataclasses.field(default_factory=list)
    policies: List[PolicyWorker] = dataclasses.field(default_factory=list)
    trainers: List[TrainerWorker] = dataclasses.field(default_factory=list)
    eval_managers: Optional[List[EvaluationManager]] = dataclasses.field(
        default_factory=list)
    master_worker: Optional[List[MasterWorker]] = dataclasses.field(
        default_factory=list)
    dummy_workers: Optional[List[DummyWorker]] = dataclasses.field(
        default_factory=list)

    def set_worker_information(self, experiment_name, trial_name):
        for worker_type, workers in [
            ("actor", self.actors),
            ("policy", self.policies),
            ("trainer", self.trainers),
            ("eval_manager", self.eval_managers),
            ("master", self.master_worker),
        ]:
            for i, worker in enumerate(workers):
                if worker_type in ("policy", "trainer", "eval_manager"):
                    policy_name = worker.policy_name
                else:
                    policy_name = None

                system_worker_info = dict(experiment_name=experiment_name,
                                          trial_name=trial_name,
                                          worker_type=worker_type,
                                          worker_index=i,
                                          worker_count=len(workers),
                                          policy_name=policy_name)
                if worker.worker_info is not None:
                    worker.worker_info.system_setup(**system_worker_info)
                else:
                    worker.worker_info = WorkerInformation(
                        **system_worker_info)


class Experiment:
    """Base class for defining the procedure of an experiment.
    """

    def initial_setup(self) -> ExperimentConfig:
        """Returns a list of workers to create when a trial of the experiment is initialized."""
        raise NotImplementedError()


def dump_config_to_yaml(config, file):
    with open(file, "w") as f:
        yaml.dump(dataclass_to_dict(config), f)


def load_config_from_yaml(file):
    with open(file, "r") as f:
        return config_to_dataclass(yaml.safe_load(f))


def dataclass_to_dict(dc):
    if isinstance(dc, (str, int, float)) or dc is None:
        pass
    elif isinstance(dc, enum.Enum):
        dc = dc.value
    elif isinstance(dc, (list, tuple)):
        dc = [dataclass_to_dict(d) for d in dc]
    elif isinstance(dc, dict):
        dc = {k: dataclass_to_dict(v) for k, v in dc.items()}
    elif dataclasses.is_dataclass(dc):
        root_name = dc.__class__.__name__
        dc = dict(config_class=root_name,
                  config_value={
                      k.name: dataclass_to_dict(getattr(dc, k.name))
                      for k in dataclasses.fields(dc)
                  })
    else:
        raise f"{dc} of type {type(dc)} cannot be parse to dict."
    return dc


def config_to_dataclass(config: Union[List, Dict]):
    if isinstance(config, (list, tuple)):
        return [config_to_dataclass(c) for c in config]
    elif isinstance(config, dict):
        if "config_class" in config.keys():
            return getattr(sys.modules[__name__], config["config_class"])(**{
                k: config_to_dataclass(v)
                for k, v in config["config_value"].items()
            })
        else:
            return config
    elif isinstance(config, (str, int, float)) or config is None:
        return config
    else:
        raise NotImplementedError(config)


ALL_EXPERIMENT_CLASSES = {}


def register_experiment(name, *cls):
    assert name not in ALL_EXPERIMENT_CLASSES, name
    ALL_EXPERIMENT_CLASSES[name] = cls


def make_experiment(name) -> List[Experiment]:
    classes = ALL_EXPERIMENT_CLASSES[name]
    return [cls() for cls in classes]
