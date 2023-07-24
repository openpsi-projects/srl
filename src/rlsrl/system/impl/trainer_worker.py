from typing import Optional, Tuple, Any
import collections
import datetime
import logging
import numpy as np
import queue
import time
import threading
import torch

from rlsrl.base.gpu_utils import set_cuda_device
from rlsrl.base.network import find_free_port
import rlsrl.api.config as config_api
import rlsrl.api.trainer as trainer_api
import rlsrl.base.buffer as buffer_api
import rlsrl.base.timeutil as timeutil
import rlsrl.base.name_resolve as name_resolve
import rlsrl.base.names as names
import rlsrl.system.api.sample_stream as sample_stream
import rlsrl.system.api.worker_base as worker_base
import rlsrl.system.api.parameter_db as db

DDP_MAX_TRAINER_STEP_DIFFERENCE = 3
PARAMETER_DB_GC_FREQUENCY_SECONDS = 300


class GPUThread:

    def __init__(self, buffer, trainer, is_master, worker_info,
                 log_frequency_seconds, log_frequency_steps,
                 push_frequency_seconds, push_frequency_steps,
                 preemption_steps, dist_kwargs):
        self.logger = logging.getLogger("gpu-thread")
        self.timing = timeutil.Timing()
        self.dist_kwargs = dist_kwargs
        self.__logging_queue = queue.Queue(8)
        self.__checkpoint_push_queue = queue.Queue(8)
        self.__replay_entry = None

        self.__buffer: buffer_api.Buffer = buffer
        self.__is_master = is_master
        self.__trainer: trainer_api.Trainer = trainer

        self.__interrupting = False
        self.__interrupt_at_step = 1e10
        self.__steps = 0
        self.__thread = threading.Thread(target=self._run, daemon=True)

        self.__preemption_steps = preemption_steps

        self.__start_time_ns = time.monotonic_ns()
        self.__logging_control = timeutil.FrequencyControl(
            frequency_seconds=log_frequency_seconds,
            frequency_steps=log_frequency_steps)
        self.__push_control = timeutil.FrequencyControl(
            frequency_seconds=push_frequency_seconds,
            frequency_steps=push_frequency_steps)

        self.__last_buffer_get_time = None

    @property
    def distributed_steps(self):
        return self.__steps

    def stats(self):
        return {}

    def is_alive(self):
        return self.__thread.is_alive()

    def start(self):
        self.__thread.start()

    def _run(self):
        set_cuda_device(self.__trainer.policy.device)
        self.__trainer.distributed(**self.dist_kwargs)
        cnt = 0
        while True:
            if self.__interrupting:
                self.__interrupt_loop()
                break
            self._run_step()
            cnt += 1
            if cnt % 100 == 0 and not self.__buffer.empty():
                total_time = sum(v for k, v in self.timing.items()
                                 if k.count('/') == 1)
                msg = "\n==========================================\n"
                for k, v in self.timing.items():
                    msg += "{} proportion: {:.3f}\n".format(k, v / total_time)
                msg += "==========================================\n"
                msg += "Total time: {:.3f} secs, total step: {}".format(
                    total_time, cnt)
                # self.logger.info(msg)

    def __interrupt_loop(self):
        self.logger.info("Entering stopping loop.")
        while self.__steps < self.__interrupt_at_step:
            if self.__replay_entry is None:
                break
            self._run_step_on_entry(self.__replay_entry)
        self.logger.info(f"Stopping at {self.__steps}!")

    def _run_step(self):
        if not self.__buffer.empty():
            with self.timing.add_time("gpu_thread/buffer_get"):
                self.__replay_entry = self.__buffer.get()
            with self.timing.add_time("gpu_thread/run_step"):
                self._run_step_on_entry(self.__replay_entry)
        else:
            with self.timing.add_time("gpu_thread/idle"):
                time.sleep(
                    0.005
                )  # to avoid locking the buffer. We should remove this line when our buffer is thread-safe.

    def _run_step_on_entry(self, replay_entry):
        with self.timing.add_time("gpu_thread/run_step/observe_metrics"):

            sample_policy_version = replay_entry.sample.average_of(
                "policy_version_steps", ignore_negative=True)
            sample_policy_version_min = replay_entry.sample.min_of(
                "policy_version_steps", ignore_negative=True)
            if sample_policy_version is None or np.isnan(
                    sample_policy_version_min
            ) or sample_policy_version_min < 0:
                self.logger.debug(
                    f"Ignored sample with version: avg {sample_policy_version}, min {sample_policy_version_min}."
                )
                return

            sample_version_difference = self.__trainer.policy.version - sample_policy_version_min
            if sample_version_difference > self.__preemption_steps:
                self.logger.debug(
                    f"Ignored sample with version: avg {sample_policy_version}, min {sample_policy_version_min} "
                    f"(current policy version {self.__trainer.policy.version})."
                )
                return

            staleness = self.__trainer.policy.version - sample_policy_version

        with self.timing.add_time("gpu_thread/run_step/trainer_step"):
            # TODO: Temporary workaround to overwrite non-numerical field `policy_name`.
            replay_entry.sample.policy_name = None
            log_entry = self.__trainer.step(replay_entry.sample)
            log_entry.stats[
                'sample_min_policy_version'] = sample_policy_version_min
            log_entry.stats[
                'sample_version_difference'] = sample_version_difference
            log_entry.stats['buffer_qsize'] = self.__buffer.qsize()
            self.__steps += 1

        with self.timing.add_time("gpu_thread/run_step/update_priorities"):
            if log_entry.priorities is not None and isinstance(
                    self.__buffer, buffer_api.PrioritizedReplayBuffer):
                self.__buffer.update_priorities(replay_entry.sampling_indices,
                                                log_entry.priorities)

        with self.timing.add_time("gpu_thread/run_step/misc"):
            samples = replay_entry.sample.length(
                0) * replay_entry.sample.length(1)

            if self.__logging_control.check(steps=samples):
                start = time.time()
                while True:
                    try:
                        _ = self.__logging_queue.get_nowait()
                    except queue.Empty:
                        break
                self.__logging_queue.put(
                    (self.__logging_control.interval_steps, log_entry),
                    block=False)
                self.logger.debug("Logged stats, took time: %.2fs",
                                  time.time() - start)
            if self.__is_master and log_entry.agree_pushing and self.__push_control.check(
            ):
                start = time.time()
                while True:
                    try:
                        _ = self.__checkpoint_push_queue.get_nowait()
                    except queue.Empty:
                        break
                self.__checkpoint_push_queue.put(
                    self.__trainer.get_checkpoint(), block=False)
                self.logger.debug("Pushed params, took time: %.2fs",
                                  time.time() - start)

    def get_step_result(
            self) -> Tuple[int, Optional[trainer_api.TrainerStepResult]]:
        """Get results of trainer step.
        Returns:
            samples: sample count of this trainer step.
            trainer_step_result: result of this trainer step.
        """
        try:
            return self.__logging_queue.get_nowait()
        except queue.Empty:
            return -1, trainer_api.TrainerStepResult({}, -1)

    def get_checkpoint(self) -> Any:
        """Get checkpoint published by the trainer.
        Returns:
            trainer_checkpoint: checkpoint to be saved/published.
        """
        try:
            return self.__checkpoint_push_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_at_step(self, stop_at_step):
        self.__interrupting = True
        self.__interrupt_at_step = stop_at_step
        self.__thread.join(timeout=60)
        if self.__thread.is_alive():
            raise RuntimeError("Failed to join GPU thread. (timeout=15s)")


class TrainerWorker(worker_base.Worker):

    def __init__(self, ctrl=None):
        super().__init__(ctrl=ctrl)
        self.timing = timeutil.Timing()
        self._timing_cnt = 0
        self.config = None
        self.policy_name = None
        self.gpu_thread = None
        self.__stream = None
        self.__buffer = None
        self.__param_db: Optional[db.ParameterDBClient] = None
        self.__ddp_env_resolved = False
        self.__is_master = False
        self.__ddp_init_address = None
        self.__ddp_rank = None
        self.__push_tagged_control = None
        self.__gc_frequency_control = timeutil.FrequencyControl(
            frequency_seconds=PARAMETER_DB_GC_FREQUENCY_SECONDS)

        # debug zerocopy
        self.consume_time = 0
        self.batch_time = 0

    def __stop_gpu_thread(self):
        """This method tells gpu thread when to stop running.
        """

        def find_safe_interrupt_step(
                my_step,
                assume_max_difference=DDP_MAX_TRAINER_STEP_DIFFERENCE):
            for i in range(my_step - assume_max_difference,
                           my_step + assume_max_difference + 1):
                if i % (assume_max_difference * 2 + 1) == 0:
                    return i + assume_max_difference + 3  # +1 should be enough, +3 is just in case.
            else:
                raise RuntimeError("This is not possible.")

        if self.gpu_thread is not None:
            curr_step = self.gpu_thread.distributed_steps
            self.logger.info(
                f"I am at step {curr_step}. "
                f"I think step difference should be no-larger than {DDP_MAX_TRAINER_STEP_DIFFERENCE}."
            )
            stop_at_step = find_safe_interrupt_step(curr_step)
            self.logger.info(f"I think we could stop at step {stop_at_step}.")
            self.gpu_thread.stop_at_step(stop_at_step)
            self.gpu_thread = None

    def __start_gpu_thread(self, trainer, dist_kwargs):
        self.gpu_thread = GPUThread(
            buffer=self.__buffer,
            trainer=trainer,
            is_master=self.__is_master,
            worker_info=self.config.worker_info,
            log_frequency_seconds=self.config.log_frequency_seconds,
            log_frequency_steps=self.config.log_frequency_steps,
            push_frequency_seconds=self.config.push_frequency_seconds,
            push_frequency_steps=self.config.push_frequency_steps,
            preemption_steps=self.__preemption_steps,
            dist_kwargs=dist_kwargs)
        self.gpu_thread.start()

    def _stats(self):
        return dict(self.gpu_thread.stats(),
                    consume_time=self.consume_time,
                    batch_time=self.batch_time)

    def _configure(self, cfg: config_api.TrainerWorker):
        self.config = cfg
        self.policy_name = cfg.policy_name
        self.__foreign_policy = cfg.foreign_policy
        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name
        self.__worker_index = str(cfg.worker_info.worker_index)

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_determinisitc
        self.__preemption_steps = cfg.preemption_steps

        self.__buffer = buffer_api.make_buffer(cfg.buffer_name,
                                               **cfg.buffer_args)
        if (not isinstance(cfg.sample_stream, str) and cfg.sample_stream.type_
                == config_api.SampleStream.Type.SHARED_MEMORY):
            self.__stream = sample_stream.make_consumer(
                cfg.sample_stream, cfg.worker_info,
                self.ctrl.spl_ctrls[cfg.sample_stream.stream_index])
        else:
            self.__stream = sample_stream.make_consumer(
                cfg.sample_stream, worker_info=cfg.worker_info)
        self.__param_db = db.make_db(cfg.parameter_db,
                                     worker_info=cfg.worker_info)

        # Reveal DDP identity of this worker to world.
        self.__reveal_ddp_identity()
        self.__ddp_env_resolved = False

        r = self.config.worker_info
        r.policy_name = self.policy_name
        return r

    def __reveal_ddp_identity(self):
        name_resolve.add_subentry(names.trainer_ddp_peer(
            self.__experiment_name, self.__trial_name, self.policy_name),
                                  self.__worker_index,
                                  keepalive_ttl=5)

    def __setup_ddp_and_gpu_thread(self):
        """Setup pytorch ddp processes, and algorithms.
        """
        self.logger.info(
            f"Setup trainer worker {self.__worker_index} for policy {self.policy_name}"
        )

        peers = list(
            sorted(
                name_resolve.get_subtree(
                    names.trainer_ddp_peer(self.__experiment_name,
                                           self.__trial_name,
                                           self.policy_name))))
        ddp_name_resolve = names.trainer_ddp_master(self.__experiment_name,
                                                    self.__trial_name,
                                                    self.policy_name)

        assert len(peers) == len(
            set(peers)), f"Duplicated trainer worker index."
        self.__world_size = len(peers)

        self.__ddp_rank = peers.index(self.__worker_index)
        if self.__ddp_rank == 0:
            import socket
            host_ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            self.__ddp_init_address = f"tcp://{host_ip}:{port}"
            name_resolve.add(ddp_name_resolve,
                             self.__ddp_init_address,
                             keepalive_ttl=15)
        else:
            try:
                self.__ddp_init_address = name_resolve.wait(ddp_name_resolve,
                                                            timeout=5)
            except TimeoutError:
                raise TimeoutError(
                    f"DDP trainer(index:{self.__worker_index}), rank {self.__ddp_rank} for policy "
                    f"{self.policy_name} wait for ddp_init_method timeout.")

        trainer = trainer_api.make(self.config.trainer, self.config.policy)
        if self.__ddp_rank == 0:
            self.__is_master = True

        if self.config.load_buffer_on_restart:
            try:
                self.__buffer = self.__param_db.get(
                    f"{self.policy_name}_buffer_{self.__worker_index}",
                    identifier="latest")['buffer']
                self.logger.info(f"Loaded saved buffer from param_db.")
            except FileNotFoundError:
                self.logger.info(f"Saved buffer not found in param_db. Skip.")

        try:
            # Loading parameters for master in sufficient for pytorch DDP.
            # Things might be different in other cases.
            checkpoint = self.__param_db.get(self.policy_name,
                                             identifier="latest")
            trainer.load_checkpoint(checkpoint)
            self.logger.info(
                f"Loaded model with tag latest. You can re-run your "
                f"experiment by deleting your saved model parameters from parameter DB."
            )
        except FileNotFoundError:
            self.__maybe_read_foreign_policy(trainer)
            if self.__is_master:
                self.logger.warning(
                    "No saved model found. This must be the first time you run this trial."
                    "DDP master is pushing the first version.")
                trainer.policy.inc_version(
                )  # Increase policy version from -1 to 0. We start training now.
                self.__param_db.push(self.policy_name,
                                     trainer.get_checkpoint(),
                                     str(trainer.policy.version))
        dist_kwargs = dict(world_size=self.__world_size,
                           rank=self.__ddp_rank,
                           init_method=self.__ddp_init_address)
        self.__start_gpu_thread(trainer, dist_kwargs=dist_kwargs)
        if self.config.push_tag_frequency_minutes is not None:
            self.__push_tagged_control = timeutil.FrequencyControl(
                frequency_seconds=self.config.push_tag_frequency_minutes * 60,
                initial_value=True)

    def __maybe_read_foreign_policy(self, trainer):
        if self.__foreign_policy is not None:
            p = self.__foreign_policy
            spec = p.param_db
            e = p.foreign_experiment_name or self.__experiment_name
            f = p.foreign_trial_name or self.__trial_name
            pn = p.foreign_policy_name or self.policy_name
            i = p.foreign_policy_identifier or "latest"

            foreign_db = db.make_db(spec=spec,
                                    worker_info=config_api.WorkerInformation(
                                        experiment_name=e, trial_name=f))
            if self.__foreign_policy.absolute_path is not None:
                checkpoint = foreign_db.get_file(
                    self.__foreign_policy.absolute_path)
                self.logger.info(
                    f"Loaded checkpoint: {self.__foreign_policy.absolute_path}"
                )
            else:
                checkpoint = foreign_db.get(name=pn, identifier=i)
                self.logger.info(
                    f"Loaded foreign parameter: {e} -> {f} -> {pn} -> {i}")
            trainer.policy.load_checkpoint(checkpoint)

    def _poll(self):
        if not self.__ddp_env_resolved:
            self.__setup_ddp_and_gpu_thread()
            self.__ddp_env_resolved = True

        if not self.gpu_thread.is_alive():
            self.__save_buffer_if_necessary()
            raise RuntimeError("Exception in trainer worker gpu thread.")

        with self.timing.add_time("trainer_worker/consume"):
            # self.__stream.consume_to(self.__buffer, max_iter=1024)
            # With a bounded iteration count, logging and checkpoint can be processed with controlled delay.
            count = self.__stream.consume_to(self.__buffer, max_iter=1024)

        with self.timing.add_time("trainer_worker/log"):
            log_step_result = None
            # Track and log training results.
            samples = 0
            batches = 0
            for _ in range(8):
                step_samples, trainer_step_result = self.gpu_thread.get_step_result(
                )
                if step_samples <= 0:
                    break
                log_step_result = worker_base.LogEntry(
                    stats=trainer_step_result.stats,
                    step=trainer_step_result.step)
                samples += step_samples
                batches += 1

        with self.timing.add_time("trainer_worker/checkpointing"):
            # Checkpoint.
            for _ in range(8):
                checkpoint = self.gpu_thread.get_checkpoint()
                if checkpoint is None:
                    break
                else:
                    ckpt = checkpoint
                    tags = []
                    if self.__push_tagged_control is not None and self.__push_tagged_control.check(
                    ):
                        tags.append("latest_tagged")
                        tags.append(
                            datetime.datetime.now().strftime("%Y%m%d_%H%M"))
                        self.logger.info("Saving a tagged policy version: %s",
                                         tags[-1])
                    self.__param_db.push(self.policy_name,
                                         ckpt,
                                         version=str(ckpt["steps"]),
                                         tags=tags)
                if self.__gc_frequency_control.check():
                    self.__param_db.gc(self.policy_name,
                                       max_untagged_version_count=10)

        self._timing_cnt += 1
        if self._timing_cnt % 100 == 0:
            total_time = sum(v for k, v in self.timing.items()
                             if k.count('/') == 1)
            msg = "\n==========================================\n"
            for k, v in self.timing.items():
                msg += "{} proportion: {:.3f}\n".format(k, v / total_time)
            msg += "==========================================\n"
            msg += "Total time: {:.3f} secs, total step: {}".format(
                total_time, self._timing_cnt)
            # self.logger.info(msg)
        return worker_base.PollResult(
            sample_count=samples,
            batch_count=batches,
            log_entry=log_step_result,
        )

    def __save_buffer_if_necessary(self):
        try:
            # Each trainer worker saves its own buffer.
            policy_version = self.__param_db.version_of(self.policy_name,
                                                        identifier="latest")
        except FileNotFoundError:
            policy_version = 0
        if self.config.save_buffer_on_exit:
            self.__param_db.push(
                f"{self.policy_name}_buffer_{self.__worker_index}",
                dict(buffer=self.__buffer, steps=policy_version),
                version=str(policy_version),
                tags="latest")
            self.logger.info(
                "Saved replay buffer in parameter db. "
                "You can load the buffer by turning on the load_buffer_on_restart option"
                " in your next run.")

    def exit(self):
        self.__save_buffer_if_necessary()
        super(TrainerWorker, self).exit()
        self.__stream.close()
        self.__stop_gpu_thread()

    def interrupt(self):
        self.__stop_gpu_thread()
        self.__stream.close()
        super(TrainerWorker, self).interrupt()
