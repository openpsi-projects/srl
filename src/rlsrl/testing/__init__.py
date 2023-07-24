"""The helper module for testing only."""
import random
import os
import sys
import time
import mock
import torch
import threading

import rlsrl.base.name_resolve as name_resolve
import rlsrl.testing.aerochess_env
import rlsrl.testing.null_trainer
import rlsrl.testing.random_policy

_IS_GITHUB_WORKFLOW = len(os.environ.get("CI", "").strip()) > 0
os.environ["MARL_CUDA_DEVICES"] = "cpu"
_DEFAULT_WAIT_NETWORK_SECONDS = 0.5 if _IS_GITHUB_WORKFLOW else 0.05
os.environ["MARL_TESTING"] = "1"

_next_port = 20000 + random.randint(
    0, 10000)  # Random port for now, should be ok most of the time.


def get_testing_port():
    """Returns a local port for testing."""
    global _next_port
    _next_port += 1
    return _next_port


def wait_network(length=_DEFAULT_WAIT_NETWORK_SECONDS):
    time.sleep(length)


def get_test_param(version=0):
    return {
        "steps": version,
        "state_dict": {
            "linear_weights": torch.randn(10, 10)
        },
    }


TESTING_RPC_NAME_RESOLVE_SERVER_PORT = get_testing_port()
name_resolve_rpc_server = name_resolve.NameResolveServer(
    port=TESTING_RPC_NAME_RESOLVE_SERVER_PORT)
thread = threading.Thread(target=name_resolve_rpc_server.run, daemon=True)
thread.start()
