from unittest import mock
import multiprocessing as mp
import numpy as np
import random
import socket
import time
import sys
import unittest

import rlsrl.testing as testing

from rlsrl.system.api.inference_stream import make_client, make_server
from rlsrl.system.impl.remote_inference import IpInferenceClient
from rlsrl.system.api.worker_control import SharedMemoryInferenceStreamCtrl
import rlsrl.api.config as config
import rlsrl.api.policy as policy
import rlsrl.base.names as names
import rlsrl.base.shared_memory as shared_memory
import rlsrl.base.name_resolve as name_resolve
import rlsrl.system.api.parameter_db as db

experiment_name = "inference_stream_test_exp"
trial_name = "inference_stream_test_run"
policy_name = "test_policy"


def make_test_server(type_="ip",
                     port=32342,
                     policy_name="test_policy",
                     batch_size=10,
                     rank=0,
                     ctrl=None):
    if type_ == 'ip':
        return make_server(
            spec=config.InferenceStream(type_=config.InferenceStream.Type.IP,
                                        stream_name=policy_name,
                                        plugin=config.IpStreamPlugin(
                                            address=f"*:{port}")))
    elif type_ == "name":
        return make_server(spec=config.InferenceStream(
            type_=config.InferenceStream.Type.NAME, stream_name=policy_name),
                           worker_info=config.WorkerInformation(
                               experiment_name=experiment_name,
                               trial_name=trial_name,
                               worker_index=rank))
    elif type_ == "shared_memory":
        return make_server(spec=config.InferenceStream(
            type_=config.InferenceStream.Type.SHARED_MEMORY,
            stream_name=policy_name),
                           worker_info=config.WorkerInformation(
                               experiment_name=experiment_name,
                               trial_name=trial_name,
                               worker_index=rank),
                           ctrl=ctrl)
    else:
        raise NotImplementedError(f"Unknown type: {type_}.")


def make_test_client(type_="ip",
                     port=None,
                     policy_name="test_policy",
                     batch_size=10,
                     rank=0,
                     foreign_policy=None,
                     accept_update_call=True,
                     ctrl=None):
    if type_ == 'ip':
        return make_client(spec=config.InferenceStream(
            type_=config.InferenceStream.Type.IP,
            stream_name=policy_name,
            plugin=config.IpStreamPlugin(address=f"localhost:{port}")), )
    elif type_ == "name":
        return make_client(spec=config.InferenceStream(
            type_=config.InferenceStream.Type.NAME, stream_name=policy_name),
                           worker_info=config.WorkerInformation(
                               experiment_name=experiment_name,
                               trial_name=trial_name,
                               worker_index=rank))
    elif type_ == "inline":
        return make_client(spec=config.InferenceStream(
            type_=config.InferenceStream.Type.INLINE,
            stream_name=policy_name + "inline",
            plugin=config.InlineInferenceStreamPlugin(
                policy=config.Policy(type_="random_policy",
                                     args=dict(action_space=10)),
                param_db=config.ParameterDB(
                    type_=config.ParameterDB.Type.FILESYSTEM),
                pull_interval_seconds=100,
                foreign_policy=foreign_policy,
                accept_update_call=accept_update_call,
                policy_name=policy_name),
        ),
                           worker_info=config.WorkerInformation(
                               experiment_name=experiment_name,
                               trial_name=trial_name,
                               worker_index=rank))
    elif type_ == "shared_memory":
        return make_client(spec=config.InferenceStream(
            type_=config.InferenceStream.Type.SHARED_MEMORY,
            stream_name=policy_name),
                           worker_info=config.WorkerInformation(
                               experiment_name=experiment_name,
                               trial_name=trial_name,
                               worker_index=rank),
                           ctrl=ctrl)
    else:
        raise NotImplementedError(f"Unknown type: {type_}.")


class IpInferenceStreamTest(unittest.TestCase):

    def setUp(self) -> None:
        socket.gethostbyname = mock.MagicMock(return_value="127.0.0.1")
        IpInferenceClient._shake_hand = mock.Mock()
        sys.modules["gfootball"] = mock.Mock()

    def test_simple_pair(self):
        port = testing.get_testing_port()

        server = make_test_server(port=port)
        client = make_test_client(port=port)

        # No request in the queue now.
        self.assertEqual(len(server.poll_requests()), 0)

        # Post two requests from the client. The server should be able to see.
        id1 = client.post_request(
            policy.RolloutRequest(obs=np.array(["foo"]),
                                  policy_state=np.array(["foo"])), )
        id2 = client.post_request(
            policy.RolloutRequest(obs=np.array(["bar"]),
                                  policy_state=np.array(["bar"])), )
        client.flush()
        testing.wait_network()
        request_batch = server.poll_requests()

        self.assertEqual(len(request_batch), 1)  # One Bundle with two requests
        self.assertEqual(request_batch[0].length(),
                         2)  # One Bundle with two requests
        self.assertEqual(
            (request_batch[0].request_id[0, 0], request_batch[0].obs[0, 0]),
            (id1, 'foo'))
        self.assertEqual(
            (request_batch[0].request_id[1, 0], request_batch[0].obs[1, 0]),
            (id2, 'bar'))

        # No reply from the server yet.
        self.assertFalse(client.is_ready([id1], None))
        self.assertFalse(client.is_ready([id2], None))

        # Reply to one request.
        server.respond(
            policy.RolloutResult(action=np.array([[24]]),
                                 client_id=np.array([[client.client_id]],
                                                    dtype=np.int32),
                                 request_id=np.array([[id1]], dtype=np.int64)))
        testing.wait_network()
        client.poll_responses()
        self.assertTrue(client.is_ready([id1], None))
        self.assertFalse(client.is_ready([id2], None))
        self.assertFalse(client.is_ready([id1, id2], None))

        # Reply to the other.
        server.respond(
            policy.RolloutResult(action=np.array([[42]]),
                                 client_id=np.array([[client.client_id]],
                                                    dtype=np.int32),
                                 request_id=np.array([[id2]], dtype=np.int64)))
        testing.wait_network()
        client.poll_responses()
        self.assertTrue(client.is_ready([id1], None))
        self.assertTrue(client.is_ready([id2], None))
        self.assertTrue(client.is_ready([id1, id2], None))

        results = client.consume_result([id1, id2], None)
        self.assertEqual(results[0].action[0], 24)
        self.assertEqual(results[1].action[0], 42)

    def test_multiple_client_single_server(self):
        port = testing.get_testing_port()

        server = make_test_server(port=port)
        client_list = [
            make_test_client(port=port),
            make_test_client(port=port),
        ]

        # send requests from two clients
        id11 = client_list[0].post_request(
            policy.RolloutRequest(obs=np.array(["foo1"]),
                                  policy_state=np.array(["foo1"])), )
        id12 = client_list[0].post_request(
            policy.RolloutRequest(obs=np.array(["foo2"]),
                                  policy_state=np.array(["foo2"])), )
        id21 = client_list[1].post_request(
            policy.RolloutRequest(obs=np.array(["bar1"]),
                                  policy_state=np.array(["bar1"])), )
        id22 = client_list[1].post_request(
            policy.RolloutRequest(obs=np.array(["bar2"]),
                                  policy_state=np.array(["bar2"])), )
        [c.flush() for c in client_list]

        testing.wait_network()
        request_bundle = server.poll_requests()

        # should have 4 requests
        self.assertEqual(len(request_bundle), 2)
        self.assertEqual(request_bundle[0].length(0), 2)
        self.assertEqual(request_bundle[1].length(0), 2)
        server.respond(
            policy.RolloutResult(
                action=np.array([[24]]),
                client_id=np.array([[client_list[0].client_id]],
                                   dtype=np.int32),
                request_id=np.array([[id11]], dtype=np.int64)))
        testing.wait_network()
        # client1: the first request is ready but not the second
        for c in client_list:
            c.poll_responses()
        self.assertTrue(client_list[0].is_ready([id11], None))
        self.assertFalse(client_list[0].is_ready([id11, id12], None))
        server.respond(
            policy.RolloutResult(
                action=np.array([[224]]),
                client_id=np.array([[client_list[0].client_id]],
                                   dtype=np.int32),
                request_id=np.array([[id12]], dtype=np.int64)))
        testing.wait_network()
        # client1: both requests are ready
        # but nothing on client2
        for c in client_list:
            c.poll_responses()
        self.assertTrue(client_list[0].is_ready([id11, id12], None))
        self.assertFalse(client_list[1].is_ready([id21, id22], None))

        server.respond(
            policy.RolloutResult(
                action=np.array([[224]]),
                client_id=np.array([[client_list[1].client_id]],
                                   dtype=np.int32),
                request_id=np.array([[id21]], dtype=np.int64)))
        testing.wait_network()
        for c in client_list:
            c.poll_responses()
        self.assertTrue(client_list[1].is_ready([id21], None))
        self.assertFalse(client_list[1].is_ready([id21, id22], None))
        server.respond(
            policy.RolloutResult(
                action=np.array([[224]]),
                client_id=np.array([[client_list[1].client_id]],
                                   dtype=np.int32),
                request_id=np.array([[id22]], dtype=np.int64)))
        testing.wait_network()
        for c in client_list:
            c.poll_responses()
        self.assertTrue(client_list[1].is_ready([id21, id22], None))


class InlineInferenceServerTest(unittest.TestCase):

    def setUp(self):
        name_resolve.reconfigure("memory")
        name_resolve.clear_subtree(names.inference_stream(experiment_name, trial_name, ""))
        self.param_db = mock.MagicMock()
        db.make_db = mock.MagicMock(return_value=self.param_db)

    def tearDown(self):
        name_resolve.clear_subtree(names.inference_stream(experiment_name, trial_name, ""))

    def test_simple_pair(self):
        server = make_test_server(type_="name")
        client = make_test_client(type_="name")

        # No request in the queue now.
        self.assertEqual(len(server.poll_requests()), 0)

        # Post two requests from the client. The server should be able to see.
        id1 = client.post_request(
            policy.RolloutRequest(obs=np.array(["foo"]),
                                  policy_state=np.array(["foo"])), )
        client.flush()
        testing.wait_network()
        request_batch = server.poll_requests()

        self.assertEqual(len(request_batch), 1)  # One Bundle with two requests
        self.assertEqual(request_batch[0].length(),
                         1)  # One Bundle with two requests
        self.assertEqual(
            (request_batch[0].request_id[0, 0], request_batch[0].obs[0, 0]),
            (id1, 'foo'))

        # No reply from the server yet.
        self.assertFalse(client.is_ready([id1], None))

        # Reply to one request.
        server.respond(
            policy.RolloutResult(action=np.array([[24]]),
                                 client_id=np.array([[client.client_id]],
                                                    dtype=np.int32),
                                 request_id=np.array([[id1]], dtype=np.int64)))
        testing.wait_network()
        client.poll_responses()
        self.assertTrue(client.is_ready([id1], None))

    def test_name_resolving_multiple(self):
        server0 = make_test_server(type_="name")
        server1 = make_test_server(type_="name")
        client0 = make_test_client(type_="name", rank=0)
        client1 = make_test_client(type_="name", rank=1)

        client0.post_request(
            policy.RolloutRequest(obs=np.array(["foo"]),
                                  policy_state=np.array(["foo"])), )
        client0.flush()
        testing.wait_network()
        n0 = len(server0.poll_requests())
        n1 = len(server1.poll_requests())
        self.assertEqual(n0 + n1, 1)

        client1.post_request(
            policy.RolloutRequest(obs=np.array(["bar"]),
                                  policy_state=np.array(["foo"])), )
        client1.flush()
        testing.wait_network()
        m0 = len(server0.poll_requests())
        m1 = len(server1.poll_requests())
        self.assertEqual(m0 + m1, 1)
        self.assertEqual(m0 + n0, 1)
        self.assertEqual(m1 + n1, 1)

    def test_set_constant(self):
        server = make_test_server(type_="name")
        client = make_test_client(type_="name")

        with self.assertRaises(Exception):
            client.get_constant("default_state")

        x = np.random.randn(10, 4)
        server.set_constant("default_state", x)
        y = client.get_constant("default_state")
        np.testing.assert_array_equal(x, y)

    def test_inline_client(self):
        ckpt = {"steps": 1, "state_dict": "null"}
        self.param_db.get = mock.MagicMock(return_value=ckpt)
        client = make_test_client("inline")
        id1 = client.post_request(
            policy.RolloutRequest(obs=np.array(["foo"]),
                                  policy_state=np.array(["foo"])), None)
        id2 = client.post_request(
            policy.RolloutRequest(obs=np.array(["bar"]),
                                  policy_state=np.array(["bar"])), None)
        self.param_db.get.assert_called_once()
        self.assertTrue(client.is_ready([id1], None))
        self.assertTrue(client.is_ready([id2], None))
        rollout_results = client.consume_result([id1, id2], None)
        self.assertEqual(len(rollout_results), 2)
        self.assertFalse(client.is_ready([id1], None))
        self.assertFalse(client.is_ready([id2], None))

    def test_load_param(self):
        client = make_test_client(
            "inline",
            foreign_policy=config.ForeignPolicy(
                foreign_experiment_name="foo",
                foreign_trial_name="bar",
                foreign_policy_name="p",
                param_db=config.ParameterDB(
                    type_=config.ParameterDB.Type.FILESYSTEM),
                foreign_policy_identifier="i"))
        self.param_db.get.assert_not_called()
        client.load_parameter()
        self.param_db.get.assert_called_once()

        pi = self.param_db.get.call_args_list[0][1]
        self.assertEqual(pi["name"], "p")
        self.assertEqual(pi["identifier"], "i")

    def test_get_file(self):
        client = make_test_client(
            "inline",
            foreign_policy=config.ForeignPolicy(
                foreign_experiment_name="foo",
                foreign_trial_name="bar",
                foreign_policy_name="p",
                param_db=config.ParameterDB(
                    type_=config.ParameterDB.Type.FILESYSTEM),
                absolute_path="pax",
                foreign_policy_identifier="i"))
        self.param_db.get.assert_not_called()
        client.load_parameter()
        self.param_db.get.assert_not_called()
        self.param_db.get_file.assert_called_once_with("pax")

    def test_foreign_db(self):
        db.make_db = mock.MagicMock()
        client = make_test_client("inline",
                                  foreign_policy=config.ForeignPolicy(
                                      foreign_experiment_name="foo",
                                      foreign_trial_name="bar",
                                      foreign_policy_name="p",
                                      foreign_policy_identifier="i"))
        wi = db.make_db.call_args_list[0][1]["worker_info"]
        self.assertEqual(wi.experiment_name, "foo")
        self.assertEqual(wi.trial_name, "bar")


class SharedMemoryInferenceStreamTest(unittest.TestCase):

    def setUp(self):
        name_resolve.reconfigure("memory")
        name_resolve.clear_subtree(names.inference_stream(experiment_name, trial_name, ""))
        name_resolve.clear_subtree(names.shared_memory(experiment_name, trial_name, ""))
        name_resolve.clear_subtree(names.pinned_shm_qsize(experiment_name, trial_name, ""))
        self.ctrl = SharedMemoryInferenceStreamCtrl(
            shared_memory.PinnedRequestSharedMemoryControl(
                experiment_name, trial_name, f"{policy_name}_inf_request"),
            shared_memory.PinnedResponseSharedMemoryControl(
                experiment_name, trial_name, f"{policy_name}_inf_response"),
        )

    def tearDown(self):
        self.ctrl.request_ctrl.close()
        self.ctrl.response_ctrl.close()
        name_resolve.clear_subtree(names.inference_stream(experiment_name, trial_name, ""))
        name_resolve.clear_subtree(names.shared_memory(experiment_name, trial_name, ""))
        name_resolve.clear_subtree(names.pinned_shm_qsize(experiment_name, trial_name, ""))

    def test_simple_pair(self):
        client = make_test_client("shared_memory", ctrl=self.ctrl)
        server = make_test_server("shared_memory", ctrl=self.ctrl)
        for _ in range(2):
            client.register_agent()

        # No request in the queue now.
        self.assertEqual(len(server.poll_requests()), 0)

        # Post two requests from the client. The server should be able to see.
        id1 = client.post_request(policy.RolloutRequest(
            obs=np.array(["foo"]),
            policy_state=np.array(["foo"]),
        ),
                                  index=0)
        id2 = client.post_request(policy.RolloutRequest(
            obs=np.array(["bar"]),
            policy_state=np.array(["bar"], ),
        ),
                                  index=1)
        client.flush()
        request_batch = server.poll_requests()
        # print(recursive_apply(request_batch[0], lambda x: x[..., 0]))

        self.assertEqual(len(request_batch), 1)  # One Bundle with two requests
        self.assertEqual(request_batch[0].length(),
                         2)  # One Bundle with two requests
        self.assertEqual(
            (request_batch[0].request_id[0, 0], request_batch[0].obs[0, 0]),
            (id1, 'foo'))
        self.assertEqual(
            (request_batch[0].request_id[1, 0], request_batch[0].obs[1, 0]),
            (id2, 'bar'))

        # No reply from the server yet.
        self.assertFalse(client.is_ready([id1], [0]))
        self.assertFalse(client.is_ready([id2], [1]))

        # Reply to one request.
        server.respond(
            policy.RolloutResult(
                action=np.array([[24]]),
                client_id=np.array([[client.client_id]], dtype=np.int32),
                request_id=np.array([[id1]], dtype=np.int64),
                buffer_index=np.array([[0]], dtype=np.int64),
            ), )
        client.poll_responses()
        self.assertTrue(client.is_ready([id1], [0]))
        self.assertFalse(client.is_ready([id2], [1]))
        self.assertFalse(client.is_ready([id1, id2], [0, 1]))

        # Reply to the other.
        server.respond(
            policy.RolloutResult(
                action=np.array([[42]]),
                client_id=np.array([[client.client_id]], dtype=np.int32),
                request_id=np.array([[id2]], dtype=np.int64),
                buffer_index=np.array([[1]], dtype=np.int64),
            ), )
        client.poll_responses()
        self.assertTrue(client.is_ready([id1], [0]))
        self.assertTrue(client.is_ready([id2], [1]))
        self.assertTrue(client.is_ready([id1, id2], [0, 1]))

        results = client.consume_result([id1, id2], [0, 1])
        self.assertEqual(results[0].action[0], 24)
        self.assertEqual(results[1].action[0], 42)

    def test_multiple_client_single_server(self):
        server = make_test_server("shared_memory", ctrl=self.ctrl)
        client_list = [
            make_test_client("shared_memory", ctrl=self.ctrl),
            make_test_client("shared_memory", ctrl=self.ctrl),
        ]
        buffer_indices = []
        for client in client_list:
            for _ in range(2):
                buffer_indices.append(client.register_agent())

        # send requests from two clients
        id11 = client_list[0].post_request(
            policy.RolloutRequest(
                obs=np.array(["foo1"]),
                policy_state=np.array(["foo1"]),
            ), buffer_indices[0])
        id12 = client_list[0].post_request(
            policy.RolloutRequest(
                obs=np.array(["foo2"]),
                policy_state=np.array(["foo2"]),
            ), buffer_indices[1])
        id21 = client_list[1].post_request(
            policy.RolloutRequest(
                obs=np.array(["bar1"]),
                policy_state=np.array(["bar1"]),
            ), buffer_indices[2])
        id22 = client_list[1].post_request(
            policy.RolloutRequest(
                obs=np.array(["bar2"]),
                policy_state=np.array(["bar2"]),
            ), buffer_indices[3])
        [c.flush() for c in client_list]

        request_bundle = server.poll_requests()

        # should have 4 requests
        self.assertEqual(len(request_bundle), 1)
        self.assertEqual(request_bundle[0].length(0), 4)
        server.respond(
            policy.RolloutResult(
                action=np.array([[24]]),
                client_id=np.array([[client_list[0].client_id]],
                                   dtype=np.int32),
                request_id=np.array([[id11]], dtype=np.int64),
                buffer_index=np.array([[buffer_indices[0]]], dtype=np.int64),
            ))

        self.assertTrue(client_list[0].is_ready([id11], [buffer_indices[0]]))
        self.assertFalse(client_list[0].is_ready([id11, id12],
                                                 buffer_indices[:2]))
        server.respond(
            policy.RolloutResult(action=np.array([[224]]),
                                 client_id=np.array(
                                     [[client_list[0].client_id]],
                                     dtype=np.int32),
                                 request_id=np.array([[id12]], dtype=np.int64),
                                 buffer_index=np.array([[buffer_indices[1]]],
                                                       dtype=np.int64)))

        self.assertTrue(client_list[0].is_ready([id11, id12],
                                                buffer_indices[:2]))
        self.assertFalse(client_list[1].is_ready([id21, id22],
                                                 buffer_indices[2:]))

        server.respond(
            policy.RolloutResult(
                action=np.array([[442], [42]]),
                client_id=np.array(
                    [[client_list[1].client_id], [client_list[1].client_id]],
                    dtype=np.int32),
                request_id=np.array([[id21], [id22]], dtype=np.int64),
                buffer_index=np.array(
                    [[buffer_indices[2]], [buffer_indices[3]]],
                    dtype=np.int64)))

        self.assertTrue(client_list[1].is_ready([id21, id22],
                                                buffer_indices[2:]))
        res1 = client_list[0].consume_result([id11, id12], buffer_indices[:2])
        self.assertEqual(res1[0].action[0], 24)
        self.assertEqual(res1[1].action[0], 224)
        res2 = client_list[1].consume_result([id21, id22], buffer_indices[2:])
        self.assertEqual(res2[0].action[0], 442)
        self.assertEqual(res2[1].action[0], 42)

    def test_concurrent_multi_client_multi_server(self):

        def client_worker(rank, ctrl, barrier):
            name_resolve.reconfigure(
                "rpc",
                address='localhost',
                port=testing.TESTING_RPC_NAME_RESOLVE_SERVER_PORT)
            client = make_test_client("shared_memory", ctrl=ctrl)
            ring_size = 2
            buffer_indices = [
                client.register_agent() for _ in range(ring_size)
            ]
            barrier.wait()

            for r, buffer_index in enumerate(buffer_indices):
                request = policy.RolloutRequest(
                    obs=np.array([f"foo{rank}{r}_0"]),
                    policy_state=np.array([f"foo{rank}{r}_0"]),
                )
                request_id = client.post_request(request, buffer_index)
                # print(f"client {rank} sent request {request_id}")
                client.flush()

            cnt = 0
            for _ in range(5):
                for r, buffer_index in enumerate(buffer_indices):
                    while not client.is_ready(_, [buffer_index]):
                        time.sleep(0.1)
                    # print(f"client {rank} request {r} ready!")
                    resp = client.consume_result(_, [buffer_index])
                    self.assertEqual(resp[0].action[0], f"foo{rank}{r}_{cnt}")

                    request = policy.RolloutRequest(
                        obs=np.array([f"foo{rank}{r}_{cnt+1}"]),
                        policy_state=np.array([f"foo{rank}{r}_0"]),
                    )
                    request_id = client.post_request(request, buffer_index)
                    # print(f"client {rank} sent request {request_id}")
                    client.flush()
                cnt += 1

        def server_worker(rank, ctrl):
            name_resolve.reconfigure(
                "rpc",
                address='localhost',
                port=testing.TESTING_RPC_NAME_RESOLVE_SERVER_PORT)
            server = make_test_server("shared_memory", ctrl=ctrl)
            while True:
                requests = server.poll_requests()
                if len(requests) == 0:
                    time.sleep(0.01)
                    continue
                self.assertEqual(len(requests), 1)
                requests = requests[0]
                time.sleep(random.random() * 0.1)
                responses = policy.RolloutResult(
                    action=requests.obs,
                    buffer_index=requests.buffer_index,
                )
                server.respond(responses)

        server_procs = [
            mp.Process(target=server_worker, args=(i, self.ctrl), daemon=True)
            for i in range(2)
        ]
        n_clients = 4
        barrier = mp.Barrier(n_clients)
        client_procs = [
            mp.Process(target=client_worker, args=(i, self.ctrl, barrier))
            for i in range(n_clients)
        ]
        for p in server_procs + client_procs:
            p.start()
        for p in client_procs:
            p.join()


if __name__ == '__main__':
    unittest.main()
