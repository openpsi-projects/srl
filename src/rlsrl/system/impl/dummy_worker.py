import time
import logging

import rlsrl.api.config
import rlsrl.base.name_resolve
import rlsrl.system.api.worker_base as worker_base

logger = logging.getLogger("DummyWorker")


class DummyWorker(worker_base.Worker):
    # dummy worker for testing
    def __init__(self, ctrl=None):
        super().__init__(ctrl=ctrl)
        self.__count = 0

    def _configure(self, config):
        self.my_key = self.my_value = config.worker_info.worker_index
        logger.info(
            f"Before dummy worker add /dummy/{self.my_key} {self.my_value}_0")
        rlsrl.base.name_resolve.add(f"/dummy/{self.my_key}",
                                    f"{self.my_value}_0")
        logger.info(
            f"After dummy worker add /dummy/{self.my_key} {self.my_value}_0")
        return config.worker_info

    def _poll(self):
        get_subtree_results = rlsrl.base.name_resolve.get_subtree("/dummy")
        get_result = rlsrl.base.name_resolve.get(f"/dummy/{self.my_key}")

        logger.info(
            f"Before dummy worker add /dummy/{self.my_key} {self.my_value}_{self.__count}"
        )
        rlsrl.base.name_resolve.add(f"/dummy/{self.my_key}",
                                    f"{self.my_value}_{self.__count}",
                                    replace=True)
        logger.info(
            f"After dummy worker add /dummy/{self.my_key} {self.my_value}_{self.__count}"
        )

        logger.info(
            f"get_subtree_results: {get_subtree_results}, get_result: {get_result}"
        )
        self.__count += 1
        time.sleep(1)
        return worker_base.PollResult(valid=True)
