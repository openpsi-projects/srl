import os
import socket
import tempfile
import mock
import numpy as np
import unittest

from rlsrl.testing import *

from rlsrl.system.impl.remote_inference import IpInferenceClient
from rlsrl.system.impl.eval_manager import EvalManager
import rlsrl.api.config as config_api
import rlsrl.api.trainer as trainer_api
import rlsrl.base.name_resolve as name_resolve
import rlsrl.base.namedarray as namedarray
import rlsrl.system.api.sample_stream as sample_stream
import rlsrl.system.api.parameter_db as parameter_db


class TestEpisodeInfo(namedarray.NamedArray):

    def __init__(
            self,
            hp: np.ndarray = np.array([0], dtype=np.float32),
            mana: np.ndarray = np.array([0], dtype=np.float32),
    ):
        super(TestEpisodeInfo, self).__init__(hp=hp, mana=mana)


def make_config(policy_name="test_policy",
                eval_stream_name="eval_test_policy",
                worker_index=0,
                worker_count=1):
    return config_api.EvaluationManager(
        eval_sample_stream=config_api.SampleStream(
            config_api.SampleStream.Type.NAME, stream_name=eval_stream_name),
        parameter_db=config_api.ParameterDB(
            config_api.ParameterDB.Type.FILESYSTEM),
        policy_name=policy_name,
        eval_tag="evaluation",
        eval_games_per_version=5,
        worker_info=config_api.WorkerInformation("test_exp", "test_run",
                                                 "trainer", worker_index,
                                                 worker_count),
    )


def random_sample_batch(version=0, hp=0, mana=0, policy_name="test_policy"):
    return trainer_api.SampleBatch(
        obs=np.random.random(size=(10, 10)),
        reward=np.random.random(size=(10, 1)),
        policy_version_steps=np.full(shape=(10, 1), fill_value=version),
        info=TestEpisodeInfo(hp=np.full(shape=(10, 1), fill_value=hp),
                             mana=np.full(shape=(10, 1), fill_value=mana)),
        info_mask=np.concatenate([np.zeros(
            (9, 1)), np.ones((1, 1))], axis=0),
        policy_name=np.full(shape=(10, 1), fill_value=policy_name))


def make_test_producer(policy_name="test_policy", rank=0):
    producer = sample_stream.make_producer(
        config_api.SampleStream(config_api.SampleStream.Type.NAME,
                                stream_name=policy_name),
        worker_info=config_api.WorkerInformation("test_exp", "test_run",
                                                 "policy", rank, 100),
    )
    return producer


class TestEvalManager(unittest.TestCase):

    def setUp(self) -> None:
        IpInferenceClient._shake_hand = mock.Mock()
        self.__tmp = tempfile.TemporaryDirectory()
        parameter_db.PytorchFilesystemParameterDB.ROOT = os.path.join(
            self.__tmp.name, "checkpoints")

        os.environ["WANDB_MODE"] = "disabled"
        socket.gethostbyname = mock.MagicMock(return_value="127.0.0.1")
        name_resolve.reconfigure("memory", log_events=True)
        name_resolve.reconfigure("memory", log_events=True)

    def tearDown(self) -> None:
        db = parameter_db.make_db(config_api.ParameterDB(
            type_=config_api.ParameterDB.Type.FILESYSTEM),
                                  worker_info=config_api.WorkerInformation(
                                      experiment_name="test_exp",
                                      trial_name="test_run",
                                  ))
        try:
            db.clear("test_policy")
        except FileNotFoundError:
            pass

    def test_loginfo(self):
        test_parameter_db = parameter_db.make_db(
            config_api.ParameterDB(
                type_=config_api.ParameterDB.Type.FILESYSTEM),
            worker_info=config_api.WorkerInformation(
                experiment_name="test_exp",
                trial_name="test_run",
            ))
        try:
            test_parameter_db.clear("test_policy")
        except FileNotFoundError:
            pass
        eval_manager = EvalManager()
        eval_manager.configure(make_config("test_policy", "eval", "metadata"))
        producer = make_test_producer(policy_name="eval")
        wait_network()
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

        for _ in range(5):
            producer.post(random_sample_batch(version=0))
        producer.flush()
        wait_network()
        # Eval manager does not accept sample until the first version is pushed.
        for _ in range(5):
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 0)
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

        test_parameter_db.push("test_policy",
                               get_test_param(version=1),
                               version="1")
        for _ in range(5):
            producer.post(random_sample_batch(version=0))
        producer.flush()
        wait_network()
        for _ in range(5):
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 0)
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

        test_parameter_db.push("test_policy", get_test_param(20), version="20")
        for _ in range(5):
            producer.post(random_sample_batch(version=1))
        producer.flush()
        wait_network()
        for _ in range(4):
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 0)
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 1)
        self.assertEqual(r.batch_count, 1)
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

        # Evaluation manager loads to version 20. 10 episodes will be logged.
        for _ in range(10):
            producer.post(random_sample_batch(version=20))
        producer.flush()
        wait_network()
        for __ in range(2):
            for _ in range(4):
                r = eval_manager._poll()
                self.assertEqual(r.sample_count, 1)
                self.assertEqual(r.batch_count, 0)
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 1)

        test_parameter_db.push("test_policy", get_test_param(50), version="50")
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)


if __name__ == '__main__':
    unittest.main()
