from unittest import mock
import numpy as np
import os
import socket
import threading
import tempfile
import unittest
import torch.distributed as dist

from rlsrl.testing import *

from rlsrl.system.impl.remote_inference import IpInferenceClient
from rlsrl.system.impl.trainer_worker import TrainerWorker, GPUThread
from rlsrl.system.api.parameter_db import PytorchFilesystemParameterDB as FilesystemParameterDB
import rlsrl.api.config as config_api
import rlsrl.api.trainer as trainer_api
import rlsrl.base.name_resolve as name_resolve
import rlsrl.system.api.sample_stream as sample_stream
import rlsrl.system.impl.trainer_worker as trainer_worker

_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "_experiments")


def make_config(policy_name="test_policy", worker_index=0, worker_count=1):
    return config_api.TrainerWorker(
        policy_name=policy_name,
        trainer="null_trainer",
        policy=config_api.Policy("random_policy", args={"action_space": 10}),
        buffer_name="simple_queue",
        log_frequency_seconds=None,
        log_frequency_steps=1,
        sample_stream=config_api.SampleStream(
            config_api.SampleStream.Type.NAME, stream_name=policy_name),
        parameter_db=config_api.ParameterDB(
            config_api.ParameterDB.Type.FILESYSTEM),
        worker_info=config_api.WorkerInformation("test_exp", "test_run",
                                                 "trainer", worker_index,
                                                 worker_count),
    )


def make_test_producer(policy_name="test_policy", rank=0):
    producer = sample_stream.make_producer(
        config_api.SampleStream(config_api.SampleStream.Type.NAME,
                                stream_name=policy_name),
        worker_info=config_api.WorkerInformation("test_exp", "test_run",
                                                 "policy", rank, 100),
    )
    return producer


class TrainWorkerTest(unittest.TestCase):

    def setUp(self):
        IpInferenceClient._shake_hand = mock.Mock()
        self.__tmp = tempfile.TemporaryDirectory()
        self.tw = []
        FilesystemParameterDB.ROOT = os.path.join(self.__tmp.name,
                                                  "checkpoints")
        trainer_worker._PUSH_PARAMS_FREQUENCY_SECONDS = 0
        trainer_worker._LOG_STATS_FREQUENCY_SECONDS = 0

        # Disable wandb
        os.environ["WANDB_MODE"] = "disabled"
        socket.gethostbyname = mock.MagicMock(return_value="127.0.0.1")
        name_resolve.reconfigure("memory", log_events=True)
        # name_resolve.reconfigure("nfs")
        name_resolve.clear_subtree('sample_stream')
        name_resolve.clear_subtree('test_exp')
        dist.init_process_group = mock.MagicMock()
        GPUThread.start = mock.Mock()  # run manually.
        GPUThread.stop_at_step = mock.Mock()  # run manually.
        threading.Thread.is_alive = mock.MagicMock(return_value=True)

    def tearDown(self) -> None:
        for t in self.tw:
            t.exit()

    def make_worker(self):
        return TrainerWorker()

    def trainer_worker(self, count=1):
        for t in self.tw:
            t.exit()
        for _ in range(count):
            self.tw.append(self.make_worker())

    def test_trainer_worker(self):
        self.trainer_worker()
        tw = self.tw[0]
        tw.name_resolve = name_resolve.make_repository("memory")

        tw.configure(make_config())
        sample_producer = make_test_producer()
        r = tw._poll()
        self.assertEqual(r.sample_count, 0)

        example_trajectory = trainer_api.SampleBatch(
            obs=np.random.random(size=(10, 1, 10)),
            reward=np.random.random(size=(10, 1, 1)),
            policy_version_steps=np.zeros(shape=(10, 1, 1)))
        sample_producer.post(example_trajectory)
        sample_producer.flush()

        wait_network()
        r = tw._poll()  # pushes to gpu worker
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)
        tw.gpu_thread._run_step()
        r = tw._poll()
        self.assertEqual(r.sample_count, 10)
        self.assertEqual(r.batch_count, 1)

        example_trajectory = trainer_api.SampleBatch(
            obs=np.random.random(size=(10, 1, 10)),
            reward=np.random.random(size=(10, 1, 1)),
            policy_version_steps=np.full(shape=(10, 1, 1), fill_value=-1))
        sample_producer.post(example_trajectory)
        sample_producer.flush()
        wait_network()
        r = tw._poll()  # pushes to gpu worker
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)
        tw.gpu_thread._run_step()
        r = tw._poll(
        )  # GPU thread refuses to train on samples with negative version.
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

        threading.Thread.is_alive = mock.MagicMock(return_value=False)
        self.assertRaises(RuntimeError, tw._poll)
        threading.Thread.is_alive = mock.MagicMock(return_value=True)
        tw.exit()

    def test_named_trainer_multiple(self):
        from rlsrl.testing.null_trainer import NullTrainer
        NullTrainer.distributed = mock.MagicMock()
        self.trainer_worker(2)
        tw1, tw2 = self.tw
        tw1.name_resolve = name_resolve.make_repository("memory")
        tw2.name_resolve = name_resolve.make_repository("memory")

        # Setting world size > 1 will case trainer workers to use ddp. However DDP cannot be tested reliably
        # on cpu-only devices. In tests we always set world_size=1.
        tw1.configure(make_config(worker_index=0, worker_count=1))
        tw2.configure(make_config(worker_index=1, worker_count=2))
        wait_network()
        tw1._poll()
        tw2._poll()
        self.assertEqual(tw1.gpu_thread.dist_kwargs["init_method"],
                         tw2.gpu_thread.dist_kwargs["init_method"])
        self.assertSetEqual(
            {
                tw1.gpu_thread.dist_kwargs["rank"],
                tw2.gpu_thread.dist_kwargs["rank"]
            }, {0, 1})
        self.assertEqual(tw1.gpu_thread.dist_kwargs["world_size"],
                         tw2.gpu_thread.dist_kwargs["world_size"])

        sample_producer1 = make_test_producer(rank=0)
        sample_producer2 = make_test_producer(rank=1)
        sample_producer3 = make_test_producer(rank=2)

        example_trajectory = trainer_api.SampleBatch(
            obs=np.random.random(size=(10, 1, 10)),
            reward=np.random.random(size=(10, 1, 1)),
            policy_version_steps=np.zeros(shape=(10, 1, 1)),
        )
        sample_producer1.post(example_trajectory)
        sample_producer1.flush()
        wait_network()
        r1 = tw1._poll().batch_count
        r2 = tw2._poll().batch_count
        self.assertEqual(r1 + r2, 0)
        wait_network()
        tw1.gpu_thread._run_step()
        tw2.gpu_thread._run_step()
        r1 = tw1._poll()
        r2 = tw2._poll()
        self.assertSetEqual({r1.batch_count, r2.batch_count}, {0, 1})
        self.assertSetEqual({r1.sample_count, r2.sample_count}, {0, 10})

        # Samples evenly distributed.
        sample_producer1.post(example_trajectory)
        sample_producer2.post(example_trajectory)
        sample_producer1.flush()
        sample_producer2.flush()
        wait_network()  # Pushes to gpu.
        tw1._poll()
        tw2._poll()
        tw1.gpu_thread._run_step()
        tw2.gpu_thread._run_step()
        r1 = tw1._poll()
        r2 = tw2._poll()
        self.assertEqual(r1.batch_count, r2.batch_count, 1)
        self.assertEqual(r1.sample_count, r2.sample_count, 10)

        sample_producer1.post(example_trajectory)
        sample_producer2.post(example_trajectory)
        sample_producer3.post(example_trajectory)
        sample_producer1.flush()
        sample_producer2.flush()
        sample_producer3.flush()
        wait_network()
        sample1 = 0
        sample2 = 0
        batch1 = 0
        batch2 = 0
        tw1.gpu_thread._run_step()
        tw2.gpu_thread._run_step()
        r1 = tw1._poll(
        )  # Consumes to GPU, twice to make sure all trajectories are trained.
        r2 = tw2._poll()
        sample1 += r1.sample_count
        batch1 += r1.batch_count
        sample2 += r2.sample_count
        batch2 += r2.batch_count
        tw1.gpu_thread._run_step()
        tw2.gpu_thread._run_step()
        r1 = tw1._poll()
        r2 = tw2._poll()
        sample1 += r1.sample_count
        batch1 += r1.batch_count
        sample2 += r2.sample_count
        batch2 += r2.batch_count
        wait_network()  # GPU returns.
        tw1.gpu_thread._run_step()
        tw2.gpu_thread._run_step()
        r1 = tw1._poll()
        r2 = tw2._poll()
        sample1 += r1.sample_count
        batch1 += r1.batch_count
        sample2 += r2.sample_count
        batch2 += r2.batch_count
        self.assertSetEqual({batch1, batch2}, {1, 2})
        self.assertSetEqual({sample1, sample2}, {10, 20})
        tw1.exit()
        tw2.exit()


if __name__ == '__main__':
    unittest.main()
