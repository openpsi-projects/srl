from typing import Optional, Union, List, Dict
import dataclasses
import logging
import multiprocessing.connection as mp_connection
import queue
import threading
import time
import wandb
import wandb.sdk.wandb_run
import os

from rlsrl.base.gpu_utils import set_cuda_device
from rlsrl.base.user import get_user_tmp
import rlsrl.api.config as config_pkg
import rlsrl.system.api.worker_control as worker_control
import rlsrl.base.timeutil as timeutil
import rlsrl.base.name_resolve as name_resolve

_WANDB_LOG_FREQUENCY_SECONDS = 10
_TERMINAL_LOG_FREQUENCY_SECONDS = 10


@dataclasses.dataclass
class LogEntry:
    """ One entry to be logged
    """
    stats: Dict  # if logged to wandb, wandb.log(stats, step=step) if step >= 0
    step: int = -1


@dataclasses.dataclass
class PollResult:
    sample_count: int
    batch_count: int
    log_entry: Optional[LogEntry] = None


class Worker:
    """The worker base class that provides general methods and entry point.

    The typical code on the worker side is:
        worker = make_worker()  # Returns instance of Worker.
        worker.run()
    and the later is standardized here as:
        while exit command is not received:
            if worker is started:
                worker.poll()
    """

    def __init__(self, ctrl: Optional[worker_control.WorkerCtrl] = None):
        """Initializes a worker.
        """
        self.__ctrl = ctrl

        # worker state
        self.__running = False
        self.__exiting = False

        self.config = None
        self.__is_configured = False

        self.logger = logging.getLogger("worker")

        self.__worker_type = None
        self.__worker_index = None
        self.__last_successful_poll_time = None

        # Monitoring related.
        self.__wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
        self.__wandb_args = None
        self.__log_wandb: bool = False

        self.__wandb_log_freq_ctrl = timeutil.FrequencyControl(
            frequency_seconds=_WANDB_LOG_FREQUENCY_SECONDS)
        self.__terminal_log_freq_ctrl = timeutil.FrequencyControl(
            frequency_seconds=_TERMINAL_LOG_FREQUENCY_SECONDS)

        self._start_time_ns = None
        self.__wait_time_seconds = 0

    def __del__(self):
        if self.__wandb_run is not None:
            self.__wandb_run.finish()

    @property
    def ctrl(self) -> worker_control.WorkerCtrl:
        return self.__ctrl

    @property
    def is_configured(self):
        return self.__is_configured

    @property
    def wandb_run(self):
        if self.__wandb_run is None:
            wandb.login()
            for _ in range(10):
                try:
                    self.__wandb_run = wandb.init(dir=get_user_tmp(),
                                                  config=self.config,
                                                  resume="allow",
                                                  **self.__wandb_args)
                    break
                except wandb.errors.UsageError as e:
                    time.sleep(5)
            else:
                raise e
        return self.__wandb_run

    def _configure(self, config) -> config_pkg.WorkerInformation:
        """Implemented by sub-classes."""
        raise NotImplementedError()

    def _poll(self) -> PollResult:
        """Implemented by sub-classes."""
        raise NotImplementedError()

    def configure(self, config):
        assert not self.__running
        self.__worker_type = config.worker_info.worker_type
        self.__worker_index = config.worker_info.worker_index

        self.logger.debug("Configuring with: %s", config)
        self.__worker_device = config.worker_info.device
        if self.__worker_device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.__worker_device

        self.__name_resolve_address = config.worker_info.name_resolve_address
        self.__name_resolve_port = config.worker_info.name_resolve_port
        if self.__worker_type != "master" and os.environ.get(
                "MARL_TESTING") != "1":
            # master worker name resolve reconfigured in self._configure()
            if self.__name_resolve_address is not None and self.__name_resolve_port is not None:
                name_resolve.reconfigure(type_="rpc",
                                         address=self.__name_resolve_address,
                                         port=self.__name_resolve_port)
            else:
                name_resolve.reconfigure(type_="memory")

        r = self._configure(config)
        self.__worker_type = r.worker_type
        self.__worker_index = r.worker_index
        self.logger = logging.getLogger(r.worker_type + "-worker")

        # config wandb logging
        self.__wandb_run = None  # This will be lazy created by self.wandb_run().
        self.__log_wandb = (self.__worker_index == 0
                            and (self.__worker_type == "trainer"
                                 or self.__worker_type == "eval_manager"))
        if r.log_wandb is not None:
            self.__log_wandb = r.log_wandb

        self.__wandb_args = dict(
            entity=r.wandb_entity,
            project=r.wandb_project or f"{r.experiment_name}",
            group=r.wandb_group or r.trial_name,
            job_type=r.wandb_job_type or f"{r.worker_type}",
            name=r.wandb_name or f"{r.policy_name or r.worker_index}",
            id=
            f"{r.experiment_name}_{r.trial_name}_{r.policy_name or 'unnamed'}_{r.worker_type}_{r.worker_index}",
            settings=wandb.Settings(start_method="fork"),
        )
        # config std output logging
        self.__log_terminal = r.log_terminal

        self.__is_configured = True
        self.logger.info(
            f"Configured {self.__worker_type} {self.__worker_index} successfully, worker device {self.__worker_device}."
        )

    def start(self):
        self.logger.info("Starting worker")
        self.__running = True

    def pause(self):
        self.logger.info("Pausing worker")
        self.__running = False

    def exit(self):
        self.logger.info("Exiting worker")
        self.__exiting = True

    def _handle_requests(self):
        if not self.__ctrl.rx.poll():
            return
        cmd, args = self.__ctrl.rx.recv()
        if cmd == "start":
            self.start()
        elif cmd == "configure":
            self.configure(*args)
        elif cmd == "pause":
            self.pause()
        elif cmd == "exit":
            self.exit()
        else:
            raise NotImplementedError(f"Unknown command {cmd}.")
        self.__ctrl.rx.send("ok")

    def run(self):
        self._start_time_ns = time.monotonic_ns()
        self.logger.info("Running worker now")
        try:
            while not self.__exiting:
                if self.__ctrl is not None:
                    self._handle_requests()
                if not self.__running:
                    time.sleep(0.05)
                    continue
                if not self.__is_configured:
                    raise RuntimeError("Worker is not configured.")

                if self.__last_successful_poll_time:
                    wait_seconds = (time.monotonic_ns() -
                                    self.__last_successful_poll_time) / 1e9
                else:
                    wait_seconds = 0.0

                r: PollResult = self._poll()

                if self.__last_successful_poll_time:
                    one_poll_elapsed_time = (
                        time.monotonic_ns() -
                        self.__last_successful_poll_time) / 1e9
                else:
                    one_poll_elapsed_time = 0.0

                if r.sample_count == r.batch_count == 0:
                    if self.__worker_type != "actor":
                        time.sleep(0.005)
                    continue

                # Record metrics.
                self.__wait_time_seconds += wait_seconds
                total_elapsed_time = (time.monotonic_ns() -
                                      self._start_time_ns) / 1e9
                self.__last_successful_poll_time = time.monotonic_ns()
                basic_stats = dict(
                    this_poll_elapsed_time=one_poll_elapsed_time,
                    total_wait_time=self.__wait_time_seconds,
                    total_elapsed_time=total_elapsed_time)
                basic_log_entry = LogEntry(stats=basic_stats)
                if r.log_entry is not None:
                    self.__log_poll_result(
                        {
                            **r.log_entry.stats,
                            **basic_stats
                        },
                        step=r.log_entry.step,
                    )
        except KeyboardInterrupt:
            self.exit()
        except Exception as e:
            raise e

    def __log_poll_result(self, stats, step):
        if self.__log_wandb and self.__wandb_log_freq_ctrl.check():
            self.wandb_run.log(stats, step=step)
        if self.__log_terminal and self.__terminal_log_freq_ctrl.check():
            self.logger.info(
                f"{self.__worker_type} {self.__worker_index} logging stats: ")
            self.__pretty_info_log(stats, step)

    def __pretty_info_log(self, stats, step):
        res = {**stats, "step": step}
        self.logger.info("=" * 40)
        for k, v in res.items():
            if isinstance(v, float):
                self.logger.info(f"{k}: {v:.3f} ;")
            else:
                self.logger.info(f"{k}: {v:d} ;")
        self.logger.info("=" * 40)


class MappingThread:
    """Wrapped of a mapping thread.
    A mapping thread gets from up_stream_queue, process data, and puts to down_stream_queue.
    """

    def __init__(self,
                 map_fn,
                 interrupt_flag,
                 upstream_queue,
                 downstream_queue: queue.Queue = None,
                 cuda_device=None):
        """Init method of MappingThread for Policy Workers.

        Args:
            map_fn: mapping function.
            interrupt_flag: main thread sets this value to True to interrupt the thread.
            upstream_queue: the queue to get data from.
            downstream_queue: the queue to put data after processing. If None, data will be discarded after processing.
        """
        self.__map_fn = map_fn
        self.__interrupt = interrupt_flag
        self.__upstream_queue = upstream_queue
        self.__downstream_queue = downstream_queue
        self.__thread = threading.Thread(target=self._run, daemon=True)
        self.__cuda_device = cuda_device

    def is_alive(self) -> bool:
        """Check whether the thread is alive.

        Returns:
            alive: True if the wrapped thread is alive, False otherwise.
        """
        return self.__interrupt or self.__thread.is_alive()

    def start(self):
        """Start the wrapped thread.
        """
        self.__thread.start()

    def join(self):
        """Join the wrapped thread.
        """
        self.__thread.join()

    def _run(self):
        if self.__cuda_device is not None:
            set_cuda_device(self.__cuda_device)
        while not self.__interrupt:
            self._run_step()

    def _run_step(self):
        try:
            data = self.__upstream_queue.get(timeout=1)
            data = self.__map_fn(data)
            if self.__downstream_queue is not None:
                self.__downstream_queue.put(data)
        except queue.Empty:
            pass

    def stop(self):
        """Stop the wrapped thread.
        """
        self.__interrupt = True
        if self.__thread.is_alive():
            self.__thread.join()
