import logging
import numpy as np
import queue
import threading
import time

import rlsrl.api.config as config_api
import rlsrl.api.policy as policy_api
import rlsrl.base.namedarray as namedarray
import rlsrl.base.timeutil as timeutil
import rlsrl.system.api.worker_base as worker_base
import rlsrl.system.api.inference_stream as inference_stream
import rlsrl.system.api.parameter_db as db


class PolicyWorker(worker_base.Worker):
    # FIXME: remove unused code and add profiling.

    def __init__(self, ctrl=None):
        super().__init__(ctrl=ctrl)

        self.experiment_name = None
        self.policy_name = None
        self.config = None
        self.__load_policy_name = None  # Name of foreign policy might different from domestic policy.
        self.__load_absolute_path = None
        self.__stream = None
        self.__policy = None
        self.__param_db = None
        self.__policy_identifier = None
        self.__pull_frequency_control = None
        self.__pull_fail_count = 0
        self.__max_pull_fails = 60
        self.__requests_buffer = []
        self.__is_paused = False
        self.__parameter_service_client = None
        self.__initialized_thread = False

        # Queues in between the data pipeline.
        self.__inference_queue = queue.Queue(1)
        self.__respond_queue = queue.Queue(1)
        self.__param_queue = queue.Queue(1)

        # The mapping threads.
        self._inference_thread = None
        self._pull_parameter_thread = None
        self.__interrupt = False

        self.__log_frequency_control = timeutil.FrequencyControl(
            frequency_seconds=10, initial_value=True)

        # inference benchmark
        self.__last_inference_time = None
        self.__inference_step_time = None
        self.__inference_interval = None

        self.__requests_buffer_lock = threading.Lock()
        self.__inferenced_pair = []
        # debug
        self.saved_count = 0

        self._is_first_pull = True

    def _configure(self, cfg: config_api.PolicyWorker):
        self.logger = logging.getLogger(f"PW{cfg.worker_info.worker_index}")
        self.config = cfg
        self.experiment_name = cfg.worker_info.experiment_name

        self.policy_name = cfg.policy_name
        self.config.worker_info.policy_name = self.policy_name

        self.__policy: policy_api.Policy = policy_api.make(cfg.policy)
        self.__policy.eval_mode()

        # Parameter DB / Policy name related.
        if self.config.foreign_policy is not None:
            p = self.config.foreign_policy
            i = self.config.worker_info
            pseudo_worker_info = config_api.WorkerInformation(
                experiment_name=p.foreign_experiment_name or i.experiment_name,
                trial_name=p.foreign_trial_name or i.trial_name)
            self.__param_db = db.make_db(p.param_db,
                                         worker_info=pseudo_worker_info)
            self.__load_absolute_path = p.absolute_path
            self.__load_absolute_path = p.absolute_path
            self.__load_policy_name = p.foreign_policy_name or cfg.policy_name
            self.__policy_identifier = p.foreign_policy_identifier or cfg.policy_identifier
        else:
            self.__param_db = db.make_db(cfg.parameter_db,
                                         worker_info=self.config.worker_info)
            self.__load_policy_name = cfg.policy_name
            self.__policy_identifier = cfg.policy_identifier

        if cfg.parameter_service_client is not None and self.__load_absolute_path is None:
            self.__parameter_service_client = db.make_client(
                cfg.parameter_service_client, self.config.worker_info)
            self.__parameter_service_client.subscribe(
                experiment_name=self.__param_db.experiment_name,
                trial_name=self.__param_db.trial_name,
                policy_name=self.__load_policy_name,
                tag=self.__policy_identifier,
                callback_fn=self._put_ckpt)

        if cfg.policy.init_ckpt_dir is not None:
            raise DeprecationWarning("Use foreign policy instead.")

        if (not isinstance(cfg.inference_stream, str)
                and cfg.inference_stream.type_
                == config_api.InferenceStream.Type.SHARED_MEMORY):
            self.__stream = inference_stream.make_server(
                cfg.inference_stream,
                self.config.worker_info,
                self.ctrl.inf_ctrls[cfg.inference_stream.stream_index],
            )
        else:
            self.__stream = inference_stream.make_server(
                cfg.inference_stream,
                self.config.worker_info,
            )

        self.__stream.set_constant("default_policy_state",
                                   self.__policy.default_policy_state)
        self.__pull_frequency_control = timeutil.FrequencyControl(
            frequency_seconds=cfg.pull_frequency_seconds,
            # If policy has a specified initial state, do not pull the
            # saved version immediately.
            initial_value=(cfg.policy.init_ckpt_dir is None))
        self.__max_pull_fails = cfg.pull_max_failures
        self.__bs = self.config.batch_size
        self.__pull_fail_count = 0

        return self.config.worker_info

    def _put_ckpt(self, checkpoint):
        while True:
            try:
                _ = self.__param_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.__param_queue.put_nowait(checkpoint)
            self.logger.debug("Reset frequency control.")
            self.__pull_frequency_control.reset_time()
        except queue.Full:
            pass

    def _inference(self, agg_requests: policy_api.RolloutRequest):
        """Run inference for batched aggregated_request.
        """
        try:
            checkpoint = self.__param_queue.get_nowait()
            self.__policy.load_checkpoint(checkpoint)
            self.logger.debug(
                f"Loaded checkpoint version: {self.__policy.version}")
        except queue.Empty:
            pass
        st = time.monotonic()
        responses = self.__policy.rollout(agg_requests)
        responses.client_id = agg_requests.client_id
        responses.request_id = agg_requests.request_id
        responses.received_time = agg_requests.received_time
        responses.buffer_index = agg_requests.buffer_index
        responses.policy_name = np.full(shape=agg_requests.client_id.shape,
                                        fill_value=self.policy_name)
        responses.policy_version_steps = np.full(
            shape=agg_requests.client_id.shape,
            fill_value=self.__policy.version)

        self.saved_count += 1
        if self.__last_inference_time is not None:
            self.__inference_interval = time.monotonic(
            ) - self.__last_inference_time
            self.__inference_step_time = time.monotonic() - st
        self.__last_inference_time = time.monotonic()

        return responses

    def _batch_step(self):
        if len(self.__requests_buffer) > 0:
            with self.__requests_buffer_lock:
                tmp = self.__requests_buffer
                self.__requests_buffer = []
            try:
                # If the inference has not started on the queued batch, make the batch larger instead of
                # initiating another batch.
                queued_requests = [self.__inference_queue.get_nowait()]
            except queue.Empty:
                queued_requests = []

            agg_request = namedarray.recursive_aggregate(
                queued_requests + tmp, lambda x: np.concatenate(x, axis=0))

            self.__inference_queue.put_nowait(agg_request[:self.__bs])
            if agg_request.length(dim=0) > self.__bs:
                with self.__requests_buffer_lock:
                    self.__requests_buffer += [agg_request[self.__bs:]]

            return len(tmp)

        return 0

    def _pull_parameter_step(self):
        # Pull parameters from server
        if self.__pull_frequency_control.check():
            self.logger.debug("Active pull.")
            while not self.__param_queue.empty():
                self.__param_queue.get()
            self.__param_queue.put(
                self.__get_checkpoint_from_db(block=self._is_first_pull))
            self._is_first_pull = False

    def _pull_parameter(self):
        while True:
            self._pull_parameter_step()
            if self.__pull_frequency_control.frequency_seconds is not None:
                time.sleep(self.__pull_frequency_control.frequency_seconds)

    def _poll(self):
        if self.__parameter_service_client is not None:
            if not self.__parameter_service_client.is_alive():
                raise RuntimeError("Exception in subscription thread.")
        if not self.__initialized_thread:
            self._inference_thread = worker_base.MappingThread(
                self._inference,
                self.__interrupt,
                self.__inference_queue,
                self.__respond_queue,
                cuda_device=self.__policy.device)
            self._pull_parameter_thread = threading.Thread(
                target=self._pull_parameter, daemon=True)
            self._inference_thread.start()
            self._pull_parameter_thread.start()
            self.__initialized_thread = True

        if not self._pull_parameter_thread.is_alive():
            raise RuntimeError("Exception in parameter pull thread.")
        if not self._inference_thread.is_alive():
            raise RuntimeError("Exception in inference thread.")

        # main thread only do polling and responding
        samples = 0
        batches = 0
        request_batch = self.__stream.poll_requests()

        with self.__requests_buffer_lock:
            self.__requests_buffer.extend(request_batch)

        try:
            responses = self.__respond_queue.get_nowait()
            self.__stream.respond(responses)
            samples += responses.length(dim=0)
            batches += 1
        except queue.Empty:
            pass

        count = self._batch_step()

        # when respond, inference threads are doing nothing
        if self.__log_frequency_control.check():
            self.logger.debug(f"Policy version: {self.__policy.version}")

        return worker_base.PollResult(sample_count=samples,
                                      batch_count=batches)

    def get_checkpoint(self):
        raise NotImplementedError(
            "Getting State Dict from policy worker not supported.")

    def load_checkpoint(self, state_dict):
        """Update parameter of the policy worker
        """
        self.__policy.load_checkpoint(state_dict)

    def __get_checkpoint_from_db(self, block=False):
        if self.__load_absolute_path is not None:
            return self.__param_db.get_file(self.__load_absolute_path)
        else:
            return self.__param_db.get(name=self.__load_policy_name,
                                       identifier=self.__policy_identifier,
                                       block=block)

    def __stop_threads(self):
        self.__interrupt = True
        self.logger.debug(f"Stopping local threads.")
        self._inference_thread.stop()
        if self.__parameter_service_client is not None:
            self.logger.debug("Stopping parameter subscription.")
            self.__parameter_service_client.stop_listening()

    def pause(self):
        super(PolicyWorker, self).pause()
        self.__is_paused = True

    def start(self):
        if self.__is_paused:
            # Set block=True to wait for trainers to push the first model when starting a new PSRO iteration.
            while not self.__param_queue.empty():
                self.__param_queue.get()
            self.__param_queue.put(
                self.__param_db.get(self.__get_checkpoint_from_db(block=True)))
            self.__is_paused = False

        if self.__parameter_service_client is not None:
            self.__parameter_service_client.start_listening()
        super(PolicyWorker, self).start()

    def exit(self):
        self.__stop_threads()
        super(PolicyWorker, self).exit()

    def interrupt(self):
        self.__stop_threads()
        super(PolicyWorker, self).interrupt()
