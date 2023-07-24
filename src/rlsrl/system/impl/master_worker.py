import logging
import threading
import time

import rlsrl.api.config
import rlsrl.base.name_resolve
import rlsrl.system.api.parameter_db as db
import rlsrl.system.api.worker_base as worker_base

logger = logging.getLogger("MasterWorker")


class MasterWorker(worker_base.Worker):
    # A master worker is a worker similar to a centralized controller on master rank node.
    # It is responsible for name resolving and parameter service.
    # The implementation is simple and only for demonstration of distributed execution of SRL.

    def _configure(self, cfg: rlsrl.api.config.MasterWorker):
        self.__name_resolve_server = rlsrl.base.name_resolve.NameResolveServer(
            cfg.port)

        self.__name_resolve_server_thread = threading.Thread(
            target=self.__name_resolve_server.run)
        self.__name_resolve_server_thread.start()
        logger.info("Master worker started RPC name resolve server thread.")

        # self.__parameter_server = db.make_server(cfg.parameter_server,
        #                                          cfg.worker_info)
        # self.__parameter_server.update_subscription()
        # self.__parameter_server.run()

        return cfg.worker_info

    def _poll(self):
        # self.__parameter_server.update_subscription()
        time.sleep(5)
        return worker_base.PollResult(sample_count=0, batch_count=0)