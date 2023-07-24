from typing import List, Any
import logging
import numpy as np
import time

import rlsrl.api.config as config_api
import rlsrl.api.policy as policy_api
import rlsrl.base.numpy_utils as numpy_utils
import rlsrl.base.namedarray as namedarray
import rlsrl.base.timeutil as timeutil
import rlsrl.system.api.parameter_db as db
import rlsrl.system.api.inference_stream as inference_stream

_INLINE_PASSIVE_PULL_FREQUENCY_SECONDS = 2
_INLINE_PULL_PARAMETER_ON_START = True

logger = logging.getLogger("InlineInferenceStream")


class InlineInferenceClient(inference_stream.InferenceClient):

    def poll_responses(self):
        pass

    def __init__(self,
                 policy,
                 policy_name,
                 param_db,
                 worker_info,
                 pull_interval,
                 policy_identifier,
                 parameter_service_client=None,
                 foreign_policy=None,
                 accept_update_call=True,
                 population=None,
                 policy_sample_probs=None):
        self.policy_name = policy_name
        self.__policy_identifier = policy_identifier
        import os
        os.environ["MARL_CUDA_DEVICES"] = "cpu"
        self.policy = policy_api.make(policy)
        self.policy.eval_mode()
        self.__logger = logging.getLogger("Inline Inference")
        self._request_count = 0
        self.__request_buffer = []
        self._response_cache = {}
        self.__pull_freq_control = timeutil.FrequencyControl(
            frequency_seconds=pull_interval,
            initial_value=_INLINE_PULL_PARAMETER_ON_START)
        self.__passive_pull_freq_control = timeutil.FrequencyControl(
            frequency_seconds=_INLINE_PASSIVE_PULL_FREQUENCY_SECONDS,
            initial_value=_INLINE_PULL_PARAMETER_ON_START,
        )
        self.__load_absolute_path = None
        self.__accept_update_call = accept_update_call
        self.__parameter_service_client = None

        # Parameter DB / Policy name related.
        if foreign_policy is not None:
            p = foreign_policy
            i = worker_info
            pseudo_worker_info = config_api.WorkerInformation(
                experiment_name=p.foreign_experiment_name or i.experiment_name,
                trial_name=p.foreign_trial_name or i.trial_name)
            self.__param_db = db.make_db(p.param_db,
                                         worker_info=pseudo_worker_info)
            self.__load_absolute_path = p.absolute_path
            self.__load_policy_name = p.foreign_policy_name or policy_name
            self.__policy_identifier = p.foreign_policy_identifier or policy_identifier
        else:
            self.__param_db = db.make_db(param_db, worker_info=worker_info)
            self.__load_policy_name = policy_name
            self.__policy_identifier = policy_identifier

        if parameter_service_client is not None and self.__load_absolute_path is None:
            self.__parameter_service_client = db.make_client(
                parameter_service_client, worker_info)
            self.__parameter_service_client.subscribe(
                experiment_name=self.__param_db.experiment_name,
                trial_name=self.__param_db.trial_name,
                policy_name=self.__load_policy_name,
                tag=self.__policy_identifier,
                callback_fn=self.policy.load_checkpoint,
                use_current_thread=True)

        self.configure_population(population, policy_sample_probs)

        self.__log_frequency_control = timeutil.FrequencyControl(
            frequency_seconds=10)

    def configure_population(self, population, policy_sample_probs):
        if population is not None:
            assert policy_sample_probs is None or len(
                policy_sample_probs
            ) == len(population), (
                f"Size of policy_sample_probs {len(policy_sample_probs)} and population {len(population)} must be the same."
            )
            self.__population = population
            if policy_sample_probs is None:
                policy_sample_probs = np.ones(
                    len(population)) / len(population)
            self.__policy_sample_probs = policy_sample_probs
        elif self.policy_name is None:
            policy_names = self.__param_db.list_names()
            if len(policy_names) == 0:
                raise ValueError(
                    "You set policy_name and population to be None, but no existing policies were found."
                )
            logger.info(f"Auto-detected population {policy_names}")
            self.__population = policy_names
            self.__policy_sample_probs = np.ones(
                len(policy_names)) / len(policy_names)
        else:
            self.__population = None
            self.__policy_sample_probs = None

    def post_request(self, request: policy_api.RolloutRequest, _=None) -> int:
        request.request_id = np.array([self._request_count], dtype=np.int64)
        req_id = self._request_count
        self.__request_buffer.append(request)
        self._request_count += 1
        self.flush()
        return req_id

    def is_ready(self, inference_ids: List[int], _=None) -> bool:
        for req_id in inference_ids:
            if req_id not in list(self._response_cache.keys()):
                return False
        return True

    def consume_result(self, inference_ids: List[int], _=None):
        return [self._response_cache.pop(req_id) for req_id in inference_ids]

    def load_parameter(self):
        """Method exposed to Actor worker so we can reload parameter when env is done.
        """
        if self.__passive_pull_freq_control.check(
        ) and self.__accept_update_call:
            # This reduces the unnecessary workload of mongodb.
            self.__load_parameter()

    def __get_checkpoint_from_db(self, block=False):
        if self.__load_absolute_path is not None:
            return self.__param_db.get_file(self.__load_absolute_path)
        else:
            return self.__param_db.get(name=self.__load_policy_name,
                                       identifier=self.__policy_identifier,
                                       block=block)

    def __load_parameter(self):
        if self.__population is None:
            policy_name = self.policy_name
        else:
            policy_name = np.random.choice(self.__population,
                                           p=self.__policy_sample_probs)
        checkpoint = self.__get_checkpoint_from_db(
            block=self.policy.version < 0)
        self.policy.load_checkpoint(checkpoint)
        self.policy_name = policy_name
        self.__logger.debug(
            f"Loaded {self.policy_name}'s parameter of version {self.policy.version}"
        )

    def flush(self):
        if self.__pull_freq_control.check():
            self.__load_parameter()

        if self.__parameter_service_client is not None:
            self.__parameter_service_client.poll()

        if self.__log_frequency_control.check():
            self.__logger.debug(f"Policy Version: {self.policy.version}")

        if len(self.__request_buffer) > 0:
            agg_req = namedarray.recursive_aggregate(self.__request_buffer,
                                                     np.stack)
            rollout_results = self.policy.rollout(agg_req)
            rollout_results.request_id = agg_req.request_id
            rollout_results.policy_version_steps = np.full(
                shape=agg_req.client_id.shape, fill_value=self.policy.version)
            rollout_results.policy_name = np.full(
                shape=agg_req.client_id.shape, fill_value=self.policy_name)
            self.__request_buffer = []
            for i in range(rollout_results.length(dim=0)):
                self._response_cache[rollout_results.request_id[
                    i, 0]] = rollout_results[i]

    def get_constant(self, name: str) -> Any:
        if name == "default_policy_state":
            return self.policy.default_policy_state
        else:
            raise NotImplementedError(name)


inference_stream.register_client(
    config_api.InferenceStream.Type.INLINE,
    lambda spec: InlineInferenceClient(
        policy=spec.policy,
        policy_name=spec.policy_name,
        param_db=spec.param_db,
        worker_info=spec.worker_info,
        pull_interval=spec.pull_interval_seconds,
        policy_identifier=spec.policy_identifier,
        foreign_policy=spec.foreign_policy,
        accept_update_call=spec.accept_update_call,
        population=spec.population,
        parameter_service_client=spec.parameter_service_client,
        policy_sample_probs=spec.policy_sample_probs,
    ),
)
