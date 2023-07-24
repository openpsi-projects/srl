from typing import Any, Tuple, Optional, List, Callable, Dict, Union
import dataclasses
import io
import json
import hashlib
import logging
import math
import os
import queue
import random
import socket
import threading
import time
import torch
import shutil
import zmq

from rlsrl.base.user import get_user_home
import rlsrl.api.config as config_api
import rlsrl.base.names as names
import rlsrl.base.name_resolve as name_resolve
import rlsrl.base.network as network

GRACE_PERIOD = 0.05  # seconds
MULTICAST_SUBSCRIPTIONS_PER_CHANNEL = 32
MULTICAST_PACKAGE_SIZE = 50 * 1024 * 1024  # Bytes
MULTICAST_SEND_INTERVAL = 0.50  # seconds
MULTICAST_METADATA_HEXLENGTH = 16
PARAMETER_CLIENT_RECV_TIMEO = 5000  # in ms
PARAMETER_SUBSCRIPTION_ID_HEXLENGTH = 16

logger = logging.getLogger("param-db")


class ParameterDBClient:
    """Defines the communication between an user and the parameter database (aka parameter server).

    Concepts:
    + policy, or policy type: a code implementing //algorithm/policy.py.
    + policy name: the trial-level unique name to identify policy parameters. The policy workers and the
      trainer workers save and load trained policies (essentially the parameters) using this name.
    + policy version: each policy, identified by the name, can have different versions as the training time
      goes on. The version can be used to load a specific frozen policy.
    + tag: a customized human-readable alias for a policy version.
    + identifier: either the raw policy version id, or a tag.
    """

    def __init__(self, experiment_name, trial_name):
        """Creates a client on a trial namespace.
        """
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.namespace = self.experiment_name + "/" + self.trial_name

    def clone(self, source_experiment_name, source_trial_name):
        """Clones all policies and their tags from another trial namespace.
        """
        raise NotImplementedError()

    def list_names(self):
        """List all policy_names with available tags.
        """
        raise NotImplementedError()

    def clear(self, name=None):
        """Removes all saved versions of a policy. If name is None, removes all policies under the current
        trial namespace.
        """
        raise NotImplementedError()

    def gc(self,
           name,
           max_untagged_version_count=None,
           max_untagged_version_ttl=None):
        """Removes policies that are outdated.
        """
        raise NotImplementedError()

    def push(self,
             name,
             checkpoint,
             version: str,
             tags: Union[None, str, List[str]] = None,
             metadata: Dict[str, Any] = None) -> str:
        """Pushes a set of new parameters to the database.

        This is typically called by the trainer to publish the newest model parameters. An optional tag can
        be added to the version; also the "latest" tag will always be added.

        Args:
            name (str): Name of the policy to push.
            checkpoint (Any): The parameters of the policy version.
            version (str): An indicator that tells the version of the checkpoint. For example it can be the number
              of gradient steps performed to get this checkpoint.
            tags (str or List[str]): An optional tag or a list of tags to associate with the policy.
            metadata (Dict[str, Any]): An optional collection of metadata associated with the policy. Those
              are useful for evaluating and selecting policy versions on demand. Examples: training steps,
              evaluation win rate, other training/testing properties, etc.
        Returns:
            version (str): The auto-generated policy version that can be used to identify the policy.
        """
        raise NotImplementedError()

    def tag(self, name, identifier, new_tag):
        """Associates a policy version with a new tag. If the tag already exists, it will be replaced with the
        provided version.

        Args:
            name (str): Name of the policy.
            identifier (str): The policy version or a tag.
            new_tag (str): The new tag to associate with the policy version.
        """
        raise NotImplementedError()

    def get(self,
            name,
            identifier="latest",
            block: bool = False,
            retry_times=60,
            mode="pytorch") -> Any:
        """Acquires a set of parameters from the database.

        This is typically called by the policy worker to update its model parameters with the
        newest.

        Args:
            name (str): Name of the policy to get.
            identifier (str): The version or tag to identify a policy with the specified name.
            block (bool): Whether the get operation is blocking.
            retry_times (int): times to retry in case of expected exceptions.
            mode: pytorch or bytes
        Returns:
            The restored parameters of the specified version of the policy.
        Raises:
            FileNotFoundError if parameter of `identifier` is not found.
        """
        raise NotImplementedError()

    def list_versions(self, name) -> List[str]:
        """Lists policy versions, sorted by time.

        Returns:
            List of versions.
        """
        raise NotImplementedError()

    def list_tags(self, name) -> List[Tuple[str, str]]:
        """Lists tagged policy versions.

        Returns:
            Pairs of (tag, version).
        """
        raise NotImplementedError()

    def has_tag(self, name, tag) -> bool:
        """Check whether a tag exists for a policy.

        Returns:
            Check result (bool).
        """
        raise NotImplementedError()

    def version_of(self, name, identifier) -> int:
        """Get the version of an identifier.
        Args:
            name: name of the policy
            identifier: identifier of the policy version.
        """
        raise NotImplementedError()

    def update_metadata(self, name: str, version: str, metadata: dict,
                        **kwargs):
        """Update the metadata of a saved parameter.
        Args:
            name: policy_name
            version: version of the parameter
            metadata: new metadata.
        No Returns.
        """
        raise NotImplementedError()


class PytorchFilesystemParameterDB(ParameterDBClient):
    ROOT = os.path.join(get_user_home(), "marl_checkpoints/")

    @dataclasses.dataclass
    class Subscription:
        topic: str
        callback: Callable

    @staticmethod
    def purge(experiment_name, trial_name, user_namespace=None):
        ckpt_dir = os.path.join(PytorchFilesystemParameterDB.ROOT,
                                user_namespace or names.USER_NAMESPACE,
                                experiment_name, trial_name)
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)

    def __init__(self, experiment_name, trial_name, user_namespace=None):
        super().__init__(experiment_name, trial_name)
        self.__workdir = os.path.join(self.ROOT, user_namespace
                                      or names.USER_NAMESPACE, experiment_name,
                                      trial_name)
        self.__last_time = None
        self.__last_index = None
        os.makedirs(self.__workdir, exist_ok=True, mode=0o775)
        self.__sub_thread = None
        self.__stop_sub = False

    def list_names(self):
        return [
            n for n in os.listdir(self.__workdir) if len(self.list_tags(n)) > 0
        ]

    def __path_of(self, name, identifier):
        return os.path.join(self.__workdir, name, identifier)

    def __is_tag(self, name, tag):
        return os.path.islink(self.__path_of(name, tag))

    def __list_all(self, name):
        return [
            f for f in os.listdir(os.path.join(self.__workdir, name))
            if not f.endswith(".tmp")
        ]

    def _has_version(self, name, version):
        return os.path.exists(self.__path_of(
            name, version)) and not self.__is_tag(name, version)

    def _has_tag(self, name, tag):
        return os.path.exists(self.__path_of(name, tag)) and self.__is_tag(
            name, tag)

    def clone(self, source_experiment_name, source_trial_name):
        raise NotImplementedError()

    def clear(self, name=None):
        if name is None:
            shutil.rmtree(self.__workdir)
        else:
            shutil.rmtree(os.path.join(self.__workdir, name))

    def _list_outdated_version(self, name, keep_count, keep_ttl):
        if keep_ttl is not None:
            raise NotImplementedError()
        has_tag = set(v for _, v in self.list_tags(name))
        untagged = [v for v in self.list_versions(name) if v not in has_tag]
        if keep_count is not None:
            return untagged[:-keep_count]
        return []

    def _remove(self, name, version):
        os.remove(self.__path_of(name, version))

    def gc(self,
           name,
           max_untagged_version_count=None,
           max_untagged_version_ttl=None):
        gv = self._list_outdated_version(name, max_untagged_version_count,
                                         max_untagged_version_ttl)
        count = 0
        for v in gv:
            self._remove(name, v)
            count += 1
        logger.debug("Removed %d outdated versions for policy %s", count, name)

    def push(self, name, checkpoint, version: str, tags=None, metadata=None):
        assert metadata is None, "Use param_db of type `METADATA` to support metadata query."
        path = self.__path_of(name, version)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        self.tag(name, version, "latest")
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                self.tag(name, version, tag)
        return version

    def __readlink(self, link_path):
        for i in range(10):
            try:
                return os.readlink(link_path)
            except FileNotFoundError as e:
                raise e
            except OSError:
                time.sleep(0.005)
        else:
            raise OSError("Read symlink Failed.")

    def tag(self, name, identifier, new_tag):
        if self.__is_tag(name, identifier):
            identifier = self.__readlink(self.__path_of(name, identifier))
        tmp_path = self.__path_of(name, new_tag + ".tmp")
        new_path = self.__path_of(name, new_tag)
        os.symlink(identifier, tmp_path)
        shutil.move(src=tmp_path, dst=new_path)

    def has_tag(self, name, identifier):
        path = self.__path_of(name, identifier)
        return os.path.lexists(path)

    def version_of(self, name, identifier) -> int:
        # TODO: add version to param_db API.
        ckpt = self.get(name, identifier)
        return ckpt.get("steps", -1)

    def get(self,
            name,
            identifier="latest",
            block=False,
            retry_times=60,
            mode="pytorch"):
        path = self.__path_of(name, identifier)
        while retry_times >= 0:
            if block and not os.path.lexists(path):
                time.sleep(1)
                retry_times -= 1
                continue
            try:
                if mode == "pytorch":
                    load_max_retry = 1 if not block else 10
                    while load_max_retry > 0:
                        try:
                            return torch.load(path, map_location="cpu")
                        except FileNotFoundError as e:
                            raise e
                        except Exception as e:
                            load_max_retry -= 1
                        time.sleep(0.1)
                elif mode == "bytes":
                    with open(path, "rb") as f:
                        return f.read()
                else:
                    raise NotImplementedError()
            except FileNotFoundError as e:
                raise e
            except OSError as e:
                time.sleep(0.005)
                logger.error(f"Read parameter failed due to {e}. Retrying...")
                retry_times -= 1
                continue
        raise FileNotFoundError(f"Read checkpoint failed {name} {identifier}.")

    @staticmethod
    def get_file(file_path):
        return torch.load(file_path, map_location="cpu")

    def list_versions(self, name):
        rs = []
        if not os.path.isdir(os.path.join(self.__workdir, name)):
            return rs
        for version in self.__list_all(os.path.join(self.__workdir, name)):
            if not self.__is_tag(name, version):
                rs.append(version)
        rs.sort(key=lambda x: int(x))  # version is number of steps.
        return rs

    def list_tags(self, name):
        rs = []
        if not os.path.isdir(os.path.join(self.__workdir, name)):
            return rs
        for tag in self.__list_all(os.path.join(self.__workdir, name)):
            if self.__is_tag(name, tag):
                rs.append((tag, self.__readlink(self.__path_of(name, tag))))
        return rs

    def update_metadata(self, name: str, version: str, metadata: dict,
                        **kwargs):
        raise NotImplementedError()


class ParameterServer:

    @property
    def serving_instances(self):
        raise NotImplementedError()

    def serve_all(self):
        """Serve all parameters.
        """
        raise NotImplementedError()

    def update_subscription(self):
        """Update serving list.
        """
        raise NotImplementedError()


class ParameterServiceClient:

    def subscribe(self,
                  experiment_name,
                  trial_name,
                  policy_name,
                  tag,
                  callback_fn,
                  use_current_thread=False,
                  *args,
                  **kwargs) -> str:
        """Subscribe to a parameter specification.
        Returns:
            subscription key: str, can be used to unsub.
        """
        raise NotImplementedError()

    @property
    def subscriptions(self):
        raise NotImplementedError()

    def is_alive(self) -> bool:
        raise NotImplementedError()

    def start_listening(self):
        raise NotImplementedError()

    def unsubscribe(self, key: str):
        raise NotImplementedError

    def stop_listening(self):
        raise NotImplementedError()

    def poll(self):
        raise NotImplementedError()

    def reset_subscriptions(self):
        raise NotImplementedError()


class MultiCastParameterServiceClient(ParameterServiceClient):
    # TODO: Complete docstrings

    class Subscription:

        def __init__(self, sub_request: Dict, host_experiment_name,
                     host_trial_name, callback: Callable, recv_timeo, rank):
            self.__param_id = param_id = json.dumps(sub_request,
                                                    separators=(',', ':'),
                                                    sort_keys=True)
            self.__hash_id = hashlib.sha256(self.__param_id.encode(
                "ascii")).hexdigest()[:PARAMETER_SUBSCRIPTION_ID_HEXLENGTH]
            self.__topic = self.__hash_id.encode("ascii")

            self.__logger = logging.getLogger("Parameter Subscription")
            self.__logger.info(f"subscribing to {param_id}")

            hostip = network.gethostip()
            self.__sub_key = name_resolve.add_subentry(
                names.parameter_subscription(host_experiment_name,
                                             host_trial_name),
                value=f"{hostip}_{param_id}",
                delete_on_exit=True,
                keepalive_ttl=10)

            self.__host_experiment_name = host_experiment_name
            self.__host_trial_name = host_trial_name
            self.__rank = rank
            self.__call_back = callback
            self.__socket = None
            self.__thread = threading.Thread(target=self.run, daemon=True)
            # flag: threading.Event = dataclasses.field(default_factory=threading.Event())
            self.__interrupt = True
            self.__reset_socket = False
            self.__socket = None
            self.__recv_timeo = recv_timeo
            self.__buffer = None
            self.__msg_sum = None
            self.__serving_idx = -1
            self.__pending_chunks = set()

        def __setup_socket(self) -> zmq.Socket:
            try:
                for _ in range(60):  # 60 trials to get hsot ip
                    server_urls = name_resolve.get_subtree(
                        names.parameter_server(
                            self.__host_experiment_name,
                            self.__host_trial_name,
                            parameter_id_str=self.__hash_id))
                    self.__logger.debug(f"Found server list: {server_urls}")
                    if len(server_urls) > 0:
                        server_url = server_urls[self.__rank %
                                                 len(server_urls)]
                        break
                    time.sleep(1)
                else:
                    raise KeyError("No parameter server found.")

                ctx = zmq.Context()
                sub_socket = ctx.socket(zmq.SUB)
                sub_socket.setsockopt(zmq.RATE, 10000000)
                sub_socket.setsockopt(zmq.MULTICAST_LOOP, 0)
                # sub_socket.setsockopt(zmq.CONFLATE, True)
                sub_socket.setsockopt(zmq.SUBSCRIBE, self.__topic)
                if server_url.startswith("epgm"):
                    hostip = socket.gethostbyname(network.gethostname())
                    server_url = server_url.format(hostip)
                sub_socket.connect(server_url)
                self.__logger.info(f"Listening on {server_url}")
                sub_socket.RCVTIMEO = self.__recv_timeo
            except TimeoutError as e:
                logger.error(
                    f"Cannot find parameter server for -e {self.__host_experiment_name} -f {self.__host_trial_name}"
                )
                raise e
            return sub_socket

        @property
        def param_id(self):
            return self.__param_id

        @property
        def socket(self):
            return self.__socket

        def is_alive(self):
            return self.__thread.is_alive()

        def reset(self):
            self.__reset_socket = True

        def stop(self):
            self.__logger.debug(f"Removing subscription key: {self.__sub_key}")
            name_resolve.delete(self.__sub_key)
            self.__interrupt = True
            if self.__thread.is_alive():
                self.__logger.debug("Joining subscription thread.")
                self.__thread.join(timeout=30)
                if self.__thread.is_alive():
                    raise RuntimeError("Failed to join subscription thread.")

        def start(self):
            self.__interrupt = False
            self.__thread.start()

        def run(self):
            if self.__socket is None:
                self.__socket = self.__setup_socket()
                self.__reset_socket = False
            while not self.__interrupt:
                if self.__reset_socket:
                    self.__socket.close(linger=0)
                    self.__socket = self.__setup_socket()
                    self.__reset_socket = False
                count = self.poll()
                if not count:
                    time.sleep(random.random())

        def poll(self):
            try:
                raw = self.__socket.recv()
                data = raw[PARAMETER_SUBSCRIPTION_ID_HEXLENGTH:]
                l = MULTICAST_METADATA_HEXLENGTH
                if len(data) < 64 + l * 5:
                    raise RuntimeError(f"Invalid message length {len(data)}")
                msg_sum = data[:64]
                chunk_idx = int(data[64:64 + l], base=16)
                chunks = int(data[64 + l:64 + l * 2], base=16)
                msg_length = int(data[64 + l * 2:64 + l * 3], base=16)
                start = int(data[64 + l * 3:64 + l * 4], base=16)
                end = int(data[64 + l * 4:64 + l * 5], base=16)
                serving_idx = int(data[64 + l * 5:64 + l * 6], base=16)
                msg = data[64 + l * 6:]

                if serving_idx < self.__serving_idx:
                    return 0

                if self.__buffer is None or msg_sum != self.__msg_sum:
                    self.__logger.debug(
                        f"Initiating new buffer, length {msg_length}, sum {msg_sum}, serving idx {serving_idx} chunks {chunks}"
                    )
                    # New message
                    self.__buffer = io.BytesIO(b"0" * msg_length)
                    self.__msg_sum = msg_sum
                    self.__pending_chunks = set(range(chunks))
                    self.__serving_idx = serving_idx

                if chunk_idx not in self.__pending_chunks or len(
                        msg) != end - start:
                    self.__logger.debug(
                        f"Unexpected chunk index {chunk_idx} out of {self.__pending_chunks}"
                    )
                else:
                    self.__buffer.getbuffer()[start:end] = msg
                    self.__pending_chunks.remove(chunk_idx)

                if len(self.__pending_chunks) == 0:
                    try:
                        total_sum = hashlib.sha256(
                            self.__buffer.getvalue()).hexdigest().encode(
                                "ascii")
                        self.__logger.debug(f"Received data sum: {total_sum}")
                        if total_sum != self.__msg_sum:
                            raise RuntimeError(
                                f"Multicast checksum failed {self.__msg_sum} != {total_sum}"
                            )

                        self.__call_back(
                            torch.load(self.__buffer, map_location="cpu"))
                    except queue.Full:
                        self.__logger.debug(
                            f"Callback failed: parameter queue is full.")
                    except Exception as e:
                        self.__logger.error(
                            f"Callback {self.__call_back} failed: {e}")
                    finally:
                        self.__buffer = None
                        return 1
                else:
                    return 0
            except zmq.ZMQError:
                return 0

    def __init__(self, experiment_name, trial_name, rank: int):
        self.__logger = logging.getLogger("MultiCast Client")
        self.__listening = False
        self.__sub_threads = []
        self.host_experiment_name = experiment_name
        self.host_trial_name = trial_name
        self.__rank = rank
        self.__subscriptions: Dict[
            int, MultiCastParameterServiceClient.Subscription] = {}

    def is_alive(self):
        return all([t.is_alive() for t in self.__subscriptions.values()])

    @property
    def subscriptions(self):
        return self.__subscriptions

    def subscribe(self,
                  experiment_name,
                  trial_name,
                  policy_name,
                  tag,
                  callback_fn,
                  use_current_thread=False,
                  *args,
                  **kwargs):
        sub_request = {
            "experiment_name": experiment_name,
            "trial_name": trial_name,
            "policy_name": policy_name,
            "tag": tag
        }
        sub = MultiCastParameterServiceClient.Subscription(
            sub_request=sub_request,
            host_experiment_name=experiment_name,
            host_trial_name=trial_name,
            callback=callback_fn,
            recv_timeo=0
            if use_current_thread else PARAMETER_CLIENT_RECV_TIMEO,
            rank=self.__rank)
        sub_id = len(self.__subscriptions.keys())
        self.__subscriptions[sub_id] = sub
        if self.__listening:
            sub.start()
        return sub_id

    def unsubscribe(self, sub_id: int):
        if sub_id not in self.__subscriptions.keys():
            raise KeyError(
                f"Cannot unsubscribe {sub_id}: No such subscription.")
        self.__logger.debug(
            f"Unsubscribing {self.__subscriptions[sub_id].param_id}")
        self.subscriptions[sub_id].stop()
        self.subscriptions.pop(sub_id)

    def start_listening(self):
        self.__logger.info("Start listening.")
        self.__listening = True
        for sub_id, param_sub in self.__subscriptions.items():
            param_sub.start()

    def stop_listening(self):
        self.__logger.debug("Stop listening.")
        self.__listening = False
        for sub_id in self.__subscriptions.keys():
            self.__subscriptions[sub_id].stop()

    def poll(self):
        if self.__listening:
            raise RuntimeError(
                "Active polling not allowed while listening on thread.")
        for sub_id, sub in self.subscriptions.items():
            if sub.socket is None:
                sub.run()
            sub.poll()

    def reset_subscriptions(self):
        for sub_id, param_sub in self.__subscriptions.items():
            param_sub.reset()


class MultiCastParameterServer(ParameterServer):
    """Multi-threaded Parameter server for multicast.
    """

    # TODO: Complete docstrings

    class ServingInstance:

        def __init__(self,
                     hash_id,
                     topic,
                     host_experiment_name,
                     host_trial_name,
                     db,
                     args,
                     serving_interval=10):
            self.__host_experiment_name = host_experiment_name
            self.__host_trial_name = host_trial_name
            self.__hash_id = hash_id
            hostname = network.gethostname()
            self.__hostip = socket.gethostbyname(hostname)
            self.__logger = logging.getLogger(
                f"Serving Thread {topic.decode('ascii')}")

            ctx = zmq.Context()
            self.__socket = ctx.socket(zmq.PUB)
            self.__socket.setsockopt(zmq.RATE, 10000000)
            self.__socket.setsockopt(zmq.MULTICAST_LOOP, 0)
            # self.__socket.setsockopt(zmq.CONFLATE, True)
            self.__topic = topic
            addr = f"tcp://{socket.gethostbyname(hostname)}"
            port = self.__socket.bind_to_random_port(addr)
            self.__addr = f"{addr}:{port}"
            self.__logger.info(f"Hosting parameter server on {self.__addr}")
            name_resolve.add_subentry(name=names.parameter_server(
                experiment_name=host_experiment_name,
                trial_name=host_trial_name,
                parameter_id_str=hash_id),
                                      value=self.__addr,
                                      keepalive_ttl=10,
                                      delete_on_exit=True)
            self.__channels = 1
            self.__db = db
            self.__args = args
            self.__serving_interval = serving_interval
            self.__interrupt = False
            self.__serving_thread = None
            self.__recent_sum = None
            self.__serving_count = 0
            self.__channel_queue = queue.Queue()

        @property
        def channels(self):
            return self.__channels

        def start(self):
            self.__interrupt = False
            self.__serving_thread = threading.Thread(target=self.run,
                                                     daemon=True)
            self.__serving_thread.start()

        def stop(self):
            self.__interrupt = True
            if self.__serving_thread is not None:
                self.__serving_thread.join(timeout=30)
                if self.__serving_thread.is_alive():
                    raise RuntimeError("Failed to join serving thread.")

        def run(self):
            while not self.__interrupt:
                pkgs = self.serve()
                clearance_time = GRACE_PERIOD
                self.__logger.debug(f"clearance time: {clearance_time}")
                self.__logger.debug(f"Multicast packages: {pkgs}")
                time.sleep(clearance_time)

        def serve(self):
            while True:
                try:
                    new_addr = self.__channel_queue.get_nowait()
                    self.__logger.debug(f"bind socket to: {new_addr}")
                    self.__socket.bind(new_addr.format(self.__hostip))
                except queue.Empty:
                    break
            msg = b""
            try:
                msg = self.__db.get(block=False, mode="bytes", **self.__args)
                check_sum = hashlib.sha256(msg).hexdigest()
                if check_sum == self.__recent_sum:
                    return 0
                self.__recent_sum = check_sum
                self.__logger.debug(
                    f"serving {self.__topic}, {self.__serving_count}, {check_sum}"
                )
                msg_length = len(msg)
                chunks = math.ceil(len(msg) / MULTICAST_PACKAGE_SIZE)
                for c in range(chunks):
                    start = c * MULTICAST_PACKAGE_SIZE
                    end = min((c + 1) * MULTICAST_PACKAGE_SIZE, msg_length)
                    multicast_pkg = self.__topic + check_sum.encode("ascii") + \
                                    "0x{0:0{1}X}".format(c, MULTICAST_METADATA_HEXLENGTH - 2).encode("ascii") + \
                                    "0x{0:0{1}X}".format(chunks, MULTICAST_METADATA_HEXLENGTH - 2).encode("ascii") + \
                                    "0x{0:0{1}X}".format(msg_length, MULTICAST_METADATA_HEXLENGTH - 2).encode("ascii") + \
                                    "0x{0:0{1}X}".format(start, MULTICAST_METADATA_HEXLENGTH - 2).encode("ascii") + \
                                    "0x{0:0{1}X}".format(end, MULTICAST_METADATA_HEXLENGTH - 2).encode("ascii") + \
                                    "0x{0:0{1}X}".format(self.__serving_count, MULTICAST_METADATA_HEXLENGTH - 2).encode("ascii") + \
                                    msg[start: end]
                    self.__socket.send(multicast_pkg)
                    time.sleep(MULTICAST_SEND_INTERVAL)
                self.__serving_count += 1
                return chunks
            except (FileNotFoundError, OSError) as e:
                self.__logger.debug(f"Failed to server {self.__topic}: {e}")
            except Exception as e:
                self.__logger.error(e)
            return len(msg)

        def add_new_channel(self):
            assert self.__addr.startswith("epgm"), self.__addr
            new_addr = f"epgm://{{}};239.192.{random.randint(1, 11)}.{random.randint(1, 255)}:{random.randint(3000, 6000)}"
            name_resolve.add_subentry(name=names.parameter_server(
                experiment_name=self.__host_experiment_name,
                trial_name=self.__host_trial_name,
                parameter_id_str=self.__hash_id),
                                      value=new_addr,
                                      delete_on_exit=True)
            self.__logger.debug(f"Adding a new channel: {new_addr}")
            if self.__serving_thread.is_alive():
                self.__channel_queue.put(new_addr)
            else:
                self.__socket.bind(new_addr.format(self.__hostip))
            self.__channels += 1

    def __init__(self, backend_db: config_api.ParameterDB, experiment_name,
                 trial_name):
        self.__logger = logging.getLogger("Parameter Server")
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.__db_type = backend_db
        self.__serving_instances: Dict[
            str, MultiCastParameterServer.ServingInstance] = {}
        self.__running = False

    def run(self):
        self.__running = True
        for hash_id, instance in self.__serving_instances.items():
            instance.start()

    def stop(self):
        self.__running = False
        for hash_id, instance in self.__serving_instances.items():
            instance.stop()

    @property
    def serving_instances(self):
        return self.__serving_instances

    def serve_all(self):
        count = 0
        for param_id, instance in self.__serving_instances.items():
            self.__logger.debug(f"serving {param_id}")
            instance.serve()
            count += 1
            time.sleep(0.5)
        return count

    def update_subscription(self):
        print("call update subscription")
        subscription_list = name_resolve.get_subtree(
            names.parameter_subscription(self.experiment_name,
                                         self.trial_name))
        print(subscription_list)
        sub_count = {k: {} for k in self.__serving_instances.keys()}
        for sub in subscription_list:
            print(sub)
            client_host, param_id = sub.split("_", 1)
            hash_id = hashlib.sha256(param_id.encode(
                "ascii")).hexdigest()[:PARAMETER_SUBSCRIPTION_ID_HEXLENGTH]
            if hash_id in self.__serving_instances.keys():
                if client_host in sub_count[hash_id].keys():
                    sub_count[hash_id][client_host] += 1
                else:
                    sub_count[hash_id][client_host] = 1
            else:
                sub_count[hash_id] = {client_host: 1}
                sub = json.loads(param_id)
                db = make_db(self.__db_type,
                             worker_info=config_api.WorkerInformation(
                                 experiment_name=sub["experiment_name"],
                                 trial_name=sub["trial_name"]))
                args = dict(name=sub["policy_name"], identifier=sub["tag"])
                self.__serving_instances[
                    hash_id] = s = MultiCastParameterServer.ServingInstance(
                        hash_id=hash_id,
                        host_experiment_name=self.experiment_name,
                        host_trial_name=self.trial_name,
                        topic=hash_id.encode("ascii"),
                        db=db,
                        args=args,
                    )
                print("add serving instance")
                if self.__running:
                    s.start()

        for hash_id, host_count in sub_count.items():
            if len(host_count) == 0:
                self.__serving_instances[hash_id].stop()
                continue

            max_subs_from_one_host = 0
            for host, count in host_count.items():
                max_subs_from_one_host = max(max_subs_from_one_host, count)
            while math.ceil(max_subs_from_one_host / MULTICAST_SUBSCRIPTIONS_PER_CHANNEL) > \
                    self.__serving_instances[hash_id].channels:
                self.__serving_instances[hash_id].add_new_channel()


def make_db(spec: config_api.ParameterDB,
            worker_info: Optional[config_api.WorkerInformation] = None):
    if spec.type_ == config_api.ParameterDB.Type.FILESYSTEM:
        return PytorchFilesystemParameterDB(
            experiment_name=worker_info.experiment_name,
            trial_name=worker_info.trial_name)
    else:
        raise NotImplementedError("Parameter db {} not implemented".format(
            spec.type_))


def make_server(spec: config_api.ParameterServer,
                worker_info: config_api.WorkerInformation) -> ParameterServer:
    if spec.type_ == config_api.ParameterServer.Type.MultiCast:
        return MultiCastParameterServer(
            backend_db=spec.backend_db,
            experiment_name=worker_info.experiment_name,
            trial_name=worker_info.trial_name)
    else:
        raise NotImplementedError()


def make_client(
        spec: config_api.ParameterServiceClient,
        worker_info: config_api.WorkerInformation) -> ParameterServiceClient:
    if spec.type_ == config_api.ParameterServiceClient.Type.MultiCast:
        return MultiCastParameterServiceClient(
            experiment_name=worker_info.experiment_name,
            trial_name=worker_info.trial_name,
            rank=worker_info.worker_index)
    else:
        raise NotImplementedError()
