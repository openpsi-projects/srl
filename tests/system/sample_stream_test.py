import sys
import mock
import multiprocessing as mp
import numpy as np
import unittest
import queue
import time

import rlsrl.testing as testing

from rlsrl.api.trainer import SampleBatch
from rlsrl.system.api.sample_stream import make_consumer, make_producer, zip_producers, NothingToConsume
import rlsrl.api.config as config
import rlsrl.base.name_resolve as name_resolve
import rlsrl.base.namedarray as namedarray
import rlsrl.base.names as names
import rlsrl.base.shared_memory as shared_memory
import rlsrl.system.impl.local_sample
import rlsrl.system.impl.remote_sample


def make_test_consumer(
    type_="ip",
    port=-1,
    experiment_name="test_exp",
    trial_name="test_run",
    policy_name="test_policy",
    rank=0,
    batch_size=2,
    qsize=None,
    ctrl=None,
):
    if type_ == "ip":
        consumer = make_consumer(
            config.SampleStream(
                type_=config.SampleStream.Type.IP,
                plugin=config.IpStreamPlugin(address=f"*:{port}")))
    elif type_ == "name":
        consumer = make_consumer(
            config.SampleStream(type_=config.SampleStream.Type.NAME,
                                stream_name=policy_name),
            config.WorkerInformation(experiment_name=experiment_name,
                                     trial_name=trial_name,
                                     worker_index=rank))
    elif type_ == "shared_memory":
        assert qsize is not None
        consumer = make_consumer(config.SampleStream(
            type_=config.SampleStream.Type.SHARED_MEMORY,
            stream_name=policy_name,
            plugin=config.SharedMemorySampleStreamPlugin(
                batch_size=batch_size,
                qsize=qsize,
            ),
        ),
                                 config.WorkerInformation(
                                     experiment_name=experiment_name,
                                     trial_name=trial_name),
                                 ctrl=ctrl)
    return consumer


def make_test_producer(type_="ip",
                       port=-1,
                       experiment_name="test_exp",
                       trial_name="test_run",
                       policy_name="test_policy",
                       rank=0,
                       qsize=None,
                       ctrl=None):
    if type_ == "ip":
        producer = make_producer(
            config.SampleStream(
                type_=config.SampleStream.Type.IP,
                plugin=config.IpStreamPlugin(address=f"localhost:{port}")))
    elif type_ == "name":
        producer = make_producer(
            config.SampleStream(type_=config.SampleStream.Type.NAME,
                                stream_name=policy_name),
            config.WorkerInformation(experiment_name=experiment_name,
                                     trial_name=trial_name,
                                     worker_index=rank))
    elif type_ == "round_robin":
        producer = make_producer(
            config.SampleStream(
                type_=config.SampleStream.Type.NAME_ROUND_ROBIN,
                stream_name=policy_name),
            config.WorkerInformation(experiment_name=experiment_name,
                                     trial_name=trial_name))
    elif type_ == "shared_memory":
        assert qsize is not None
        producer = make_producer(config.SampleStream(
            type_=config.SampleStream.Type.SHARED_MEMORY,
            stream_name=policy_name,
            plugin=config.SharedMemorySampleStreamPlugin(qsize=qsize),
        ),
                                 config.WorkerInformation(
                                     experiment_name=experiment_name,
                                     trial_name=trial_name),
                                 ctrl=ctrl)
    return producer


def run_shared_memory_producer():
    pass


def run_shared_memory_consumer():
    pass


def make_sample_batch(sample_steps, version=0):
    return namedarray.recursive_aggregate([
        SampleBatch(
            obs=np.random.random((10, )).astype(np.float32),
            policy_state=np.random.random((2, 2)).astype(np.float32),
            on_reset=np.array([False], dtype=np.uint8),
            action=np.array([np.random.randint(19)]).astype(np.int32),
            log_probs=np.random.random(1, ).astype(np.int32),
            reward=np.array([0], dtype=np.float32).astype(np.int32),
            info=np.random.randint(0, 2, (1, )),
            policy_version_steps=np.array([version], dtype=np.int64),
        ) for _ in range(sample_steps)
    ], np.stack)


class SampleStreamTest(unittest.TestCase):

    def setUp(self):
        name_resolve.reconfigure("memory", log_events=True)
        name_resolve.clear_subtree("/")
        sys.modules["gfootball"] = mock.Mock()
        sys.modules["gfootball.env"] = mock.Mock()

    def tearDown(self):
        name_resolve.clear_subtree("/")

    def test_simple(self):
        port = testing.get_testing_port()
        consumer = make_test_consumer(port=port)
        producer = make_test_producer(port=port)
        buffer = queue.Queue()

        self.assertEqual(consumer.consume_to(buffer), 0)
        self.assertTrue(buffer.empty())

        producer.post(make_sample_batch(5))
        producer.flush()
        testing.wait_network()
        self.assertEqual(consumer.consume_to(buffer), 1)
        self.assertFalse(buffer.empty())
        self.assertEqual(buffer.get().length(dim=0), 5)
        self.assertTrue(buffer.empty())

        producer.post(namedarray.from_dict(dict(a=np.array([5, 6, 7]))))
        producer.post(namedarray.from_dict(dict(a=np.array([3, 3, 3]))))
        producer.flush()
        testing.wait_network()
        self.assertEqual(consumer.consume_to(buffer), 2)
        self.assertFalse(buffer.empty())
        np.testing.assert_equal(buffer.get().a, [5, 6, 7])
        np.testing.assert_equal(buffer.get().a, [3, 3, 3])
        self.assertTrue(buffer.empty())

    def test_multiple_producers(self):
        port = testing.get_testing_port()
        consumer = make_test_consumer(port=port)
        producer1 = make_test_producer(port=port)
        producer2 = make_test_producer(port=port)
        buffer = queue.Queue()

        self.assertEqual(consumer.consume_to(buffer), 0)
        self.assertTrue(buffer.empty())

        producer1.post(make_sample_batch(5))
        producer2.post(make_sample_batch(6))
        producer1.flush()
        producer2.flush()
        testing.wait_network()
        self.assertEqual(consumer.consume_to(buffer), 2)
        self.assertFalse(buffer.empty())
        self.assertSetEqual(
            {buffer.get().length(dim=0),
             buffer.get().length(dim=0)}, {5, 6})  # Order is not guaranteed.
        self.assertTrue(buffer.empty())

        producer1.post(make_sample_batch(5))
        producer1.post(make_sample_batch(5))
        producer2.post(make_sample_batch(5))
        producer1.post(make_sample_batch(5))
        producer1.flush()
        producer2.flush()
        time.sleep(0.01)
        self.assertEqual(consumer.consume_to(buffer), 4)
        self.assertFalse(buffer.empty())
        [buffer.get() for _ in range(4)]
        self.assertTrue(buffer.empty())

    def test_name_resolving_pair(self):
        consumer = make_test_consumer(type_="name")
        producer = make_test_producer(type_="name", rank=0)
        buffer = queue.Queue()

        self.assertEqual(consumer.consume_to(buffer), 0)
        self.assertTrue(buffer.empty())

        producer.post(make_sample_batch(5))
        producer.flush()
        testing.wait_network()
        self.assertEqual(consumer.consume_to(buffer), 1)
        self.assertFalse(buffer.empty())
        self.assertEqual(buffer.get().length(0), 5)
        self.assertTrue(buffer.empty())

        producer.post(make_sample_batch(5))
        producer.post(make_sample_batch(5))
        producer.flush()
        testing.wait_network()
        self.assertEqual(consumer.consume_to(buffer), 2)
        self.assertFalse(buffer.empty())
        buffer.get()
        buffer.get()
        self.assertTrue(buffer.empty())

    def test_name_resolving_multiple(self):
        consumer1 = make_test_consumer(type_="name")
        consumer2 = make_test_consumer(type_="name")
        if consumer1.address > consumer2.address:
            consumer1, consumer2 = consumer2, consumer1
        producer1 = make_test_producer(type_="name", rank=0)
        producer2 = make_test_producer(type_="name", rank=1)
        buffer = queue.Queue()

        self.assertEqual(consumer1.consume_to(buffer), 0)
        self.assertEqual(consumer2.consume_to(buffer), 0)
        self.assertTrue(buffer.empty())

        producer1.post(make_sample_batch(5))
        producer2.post(make_sample_batch(6))
        producer1.flush()
        producer2.flush()
        testing.wait_network()
        self.assertEqual(consumer1.consume_to(buffer), 1)
        self.assertFalse(buffer.empty())
        self.assertEqual(buffer.get().length(dim=0), 5)
        self.assertEqual(consumer2.consume_to(buffer), 1)
        self.assertFalse(buffer.empty())
        self.assertEqual(buffer.get().length(dim=0), 6)

        self.assertTrue(buffer.empty())

        producer1.post(make_sample_batch(5))
        producer1.post(make_sample_batch(5))
        producer2.post(make_sample_batch(5))
        producer1.post(make_sample_batch(5))
        producer2.post(make_sample_batch(5))
        producer1.flush()
        producer2.flush()

        testing.wait_network()
        self.assertEqual(consumer1.consume_to(buffer), 3)
        self.assertEqual(consumer2.consume_to(buffer), 2)
        self.assertFalse(buffer.empty())
        [buffer.get() for _ in range(5)]
        self.assertTrue(buffer.empty())

    def test_zip(self):
        consumer1 = make_test_consumer(type_="name", policy_name="alice")
        consumer2 = make_test_consumer(type_="name", policy_name="bob")
        producer1 = make_test_producer(type_="name", policy_name="alice")
        producer1.post = mock.MagicMock()
        producer2 = make_test_producer(type_="name", policy_name="bob")
        producer2.post = mock.MagicMock()
        zipped_producer = zip_producers([producer1, producer2])
        zipped_producer.post(make_sample_batch(5, version=2))
        zipped_producer.flush()
        producer1.post.assert_called_once()
        producer2.post.assert_called_once()

    def test_round_robin(self):
        buffer = queue.Queue()
        consumer1 = make_test_consumer(type_="name", policy_name="alice")
        consumer2 = make_test_consumer(type_="name", policy_name="alice")
        producer = make_test_producer(type_="round_robin", policy_name="alice")
        producer.post(make_sample_batch(5))
        producer.flush()
        testing.wait_network()
        c11 = consumer1.consume_to(buffer)
        c21 = consumer2.consume_to(buffer)
        producer.post(make_sample_batch(5))
        producer.flush()
        testing.wait_network()
        c12 = consumer1.consume_to(buffer)
        c22 = consumer2.consume_to(buffer)
        self.assertEqual(c11 + c21, 1)
        self.assertEqual(c11 + c12, 1)
        self.assertEqual(c21 + c22, 1)
        self.assertEqual(c22 + c12, 1)

    def test_consumer_consume(self):
        consumer1 = make_test_consumer(type_="name", policy_name="alice")
        producer = make_test_producer(type_="name", policy_name="alice")
        s = make_sample_batch(5)
        producer.post(s)
        producer.flush()
        testing.wait_network()
        s1 = consumer1.consume()
        for key in s.keys():
            if s[key] is None:
                self.assertIsNone(s1[key])
            else:
                np.testing.assert_equal(s[key], s1[key])
        self.assertRaises(NothingToConsume, consumer1.consume)


class SharedMemorySampleStreamTest(unittest.TestCase):

    def setUp(self):
        name_resolve.reconfigure("memory")
        self.experiment_name = 'shm_ss_run'
        self.trial_name = 'shm_ss_trial'
        self.policy_name = 'default'
        name_resolve.clear_subtree(
            names.shared_memory(self.experiment_name, self.trial_name, ""))

    def tearDown(self):
        name_resolve.clear_subtree(
            names.shared_memory(self.experiment_name, self.trial_name, ""))

    def test_simple_pair(self):
        ctrl = shared_memory.OutOfOrderSharedMemoryControl(
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            ctrl_name=f'{self.policy_name}_samplectrl',
            reuses=3,
            qsize=10,
        )
        consumer = make_test_consumer(
            type_="shared_memory",
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            policy_name=self.policy_name,
            batch_size=5,
            ctrl=ctrl,
            qsize=10,
        )
        producer = make_test_producer(
            type_="shared_memory",
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            policy_name=self.policy_name,
            ctrl=ctrl,
            qsize=10,
        )
        buffer = queue.Queue()

        for _ in range(10):
            producer.post(make_sample_batch(5))
        with self.assertRaises(NothingToConsume):
            consumer.consume()
        self.assertEqual(consumer.consume_to(buffer), 0)

        producer.flush()
        self.assertEqual(consumer.consume_to(buffer), 6)

        sample = buffer.get()
        self.assertEqual(sample.on_reset.shape, (5, 5, 1))
        consumer.close()
        producer.close()
        ctrl.close()

    def test_many_to_many(self):
        num_consumers = 2
        num_producers = 6
        ctrl = shared_memory.OutOfOrderSharedMemoryControl(
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            ctrl_name=f'{self.policy_name}_samplectrl',
            reuses=3,
            qsize=30,
        )
        consumers = [
            make_test_consumer(
                type_="shared_memory",
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                policy_name=self.policy_name,
                batch_size=6,
                ctrl=ctrl,
                qsize=30,
            ) for _ in range(num_consumers)
        ]
        producers = [
            make_test_producer(
                type_="shared_memory",
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                policy_name=self.policy_name,
                ctrl=ctrl,
                qsize=30,
            ) for _ in range(num_producers)
        ]

        # 24 samples in total
        for p in producers:
            for _ in range(4):
                p.post(make_sample_batch(5))
            p.flush()

        # 24 samples * 3 reuses / 2 consumers / batch size 6
        for _ in range(6):
            for c in consumers:
                c.consume()

        with self.assertRaises(NothingToConsume):
            consumers[0].consume()
        with self.assertRaises(NothingToConsume):
            consumers[1].consume()

        for c in consumers:
            c.close()
        for p in producers:
            p.close()
        ctrl.close()

    def test_concurrent_many_to_many(self):
        num_consumers = 2
        num_producers = 6
        qsize = 30
        num_samples = 40
        batch_size = 24
        assert num_samples * num_producers % batch_size == 0

        ctrl = shared_memory.OutOfOrderSharedMemoryControl(
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            ctrl_name=f'{self.policy_name}_samplectrl',
            reuses=1,
            qsize=qsize,
        )

        def producer_worker(rank, ctrl, barrier):
            name_resolve.reconfigure(
                "rpc",
                address="localhost",
                port=testing.TESTING_RPC_NAME_RESOLVE_SERVER_PORT)
            producer = make_test_producer(
                type_="shared_memory",
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                policy_name=self.policy_name,
                ctrl=ctrl,
                qsize=qsize,
            )
            barrier.wait()
            for _ in range(num_samples):
                producer.post(make_sample_batch(5))
                producer.flush()
            time.sleep(1)
            producer.close()
            name_resolve.clear_subtree(
                names.shared_memory(self.experiment_name, self.trial_name, ""))

        def consumer_worker(rank, ctrl, shm_value):
            name_resolve.reconfigure(
                "rpc",
                address="localhost",
                port=testing.TESTING_RPC_NAME_RESOLVE_SERVER_PORT)
            consumer = make_test_consumer(
                type_="shared_memory",
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                policy_name=self.policy_name,
                batch_size=batch_size,
                ctrl=ctrl,
                qsize=qsize,
            )
            buffer = queue.Queue()
            for _ in range(1000):
                consumer.consume_to(buffer)
                time.sleep(
                    0.001)  # make sure that all samples will be received
            shm_value.value += buffer.qsize()
            consumer.close()
            name_resolve.clear_subtree(
                names.shared_memory(self.experiment_name, self.trial_name, ""))

        barrier = mp.Barrier(num_producers)
        shm_values = [mp.Value("i", 0) for _ in range(num_consumers)]

        procs = [
            mp.Process(target=producer_worker, args=(i, ctrl, barrier))
            for i in range(num_producers)
        ]
        procs += [
            mp.Process(target=consumer_worker, args=(j, ctrl, shm_values[j]))
            for j in range(num_consumers)
        ]

        for p in procs:
            p.start()
        for p in procs:
            p.join()
        self.assertEqual(sum(v.value for v in shm_values),
                         num_samples * num_producers // batch_size)
        ctrl.close()


if __name__ == '__main__':
    unittest.main()
