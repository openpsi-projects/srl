from typing import Tuple, List
import argparse
import itertools
import logging
import multiprocessing
import multiprocessing.connection as mp_connection
import os

# multiprocessing.set_start_method("spawn", force=True)

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT,
                    datefmt=DATE_FORMAT,
                    level=os.environ.get("LOGLEVEL", "INFO"))

import rlsrl.api.config as config_api
import rlsrl.system as system
import rlsrl.system.api.worker_control as worker_control

# Import runnable legacy code.
import rlsrl.legacy.algorithm
import rlsrl.legacy.environment
import rlsrl.legacy.experiments

logger = logging.getLogger("SRL")


def submit_workers(worker_type, ctrls, experiment_name, trial_name,
                   env_vars):
    count = len(ctrls)
    logger.info(f"Submitted {count} {worker_type} worker(s).")
    ps = [
        multiprocessing.Process(
            target=system.run_worker,
            args=(
                worker_type,
                experiment_name,
                trial_name,
                f"{worker_type}_{i}",
                ctrl,
                env_vars,
            ),
        ) for i, ctrl in enumerate(ctrls)
    ]
    for p in ps:
        p.start()
    return ps


def group_request(cmd: str, cmd_args: List[Tuple],
                  ctrl_handles: List[mp_connection.Connection]):
    for tx, arg in zip(ctrl_handles, cmd_args):
        tx.send((cmd, arg))
    return [tx.recv() for tx in ctrl_handles]


def run_local(args):
    exps = config_api.make_experiment(args.experiment_name)
    if len(exps) > 1:
        # TODO: add support for consecutive multiple experiments.
        raise NotImplementedError()

    exp: config_api.Experiment = exps[0]
    setup = exp.initial_setup()
    setup.set_worker_information(args.experiment_name, args.trial_name)
    # Filter and only remain the workers that are assigned to this node.
    for worker_type, spec in system.RL_WORKERS.items():
        field_name = spec.config_field_name
        worker_setups = getattr(setup, field_name)
        filtered = filter(lambda x: x.worker_info.node_rank == args.node_rank,
                          worker_setups)
        setattr(setup, field_name, list(filtered))

    logger.info(f"Node {args.node_rank}: Running {exp.__class__.__name__} "
                f"experiment_name: {args.experiment_name}"
                f" trial_name {args.trial_name}")

    name_resolve_address = None
    name_resolve_port = None

    workers = dict()
    worker_tx = dict()

    env_vars = {
        "PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
        "NCCL_IB_DISABLE": "1",
        "WANDB_MODE": args.wandb_mode,
        "LOGLEVEL": os.environ.get("LOGLEVEL", "INFO"),
    }

    master_setup = setup.master_worker
    if len(master_setup) > 1:
        raise RuntimeError("Only one or zero master worker is supported.")
    if len(master_setup) == 1:
        master_setup = master_setup[0]
        name_resolve_address = master_setup.address
        name_resolve_port = master_setup.port
        master_setup.worker_info.set_name_resolve_address(
            name_resolve_address, name_resolve_port)
        assert master_setup.worker_info.node_rank == 0, "Master worker should be allocated at master node (rank = 0)."
        if args.node_rank == 0:
            tx, rx = multiprocessing.Pipe()
            procs = submit_workers("master",
                                   [worker_control.WorkerCtrl(rx, None, None)],
                                   args.experiment_name, args.trial_name,
                                   env_vars)
            workers['master'] = procs
            worker_tx['master'] = [tx]

            group_request("configure", [(master_setup, )], worker_tx['master'])
            group_request("start", [()], worker_tx['master'])

    setup, ctrls, tx_handles = worker_control.make_worker_control(
        args.experiment_name, args.trial_name, setup)
    worker_tx.update(tx_handles)

    # Submit workers.
    for worker_type in system.RL_WORKERS:
        if worker_type not in ctrls:
            continue
        procs = submit_workers(worker_type, ctrls[worker_type],
                               args.experiment_name, args.trial_name, env_vars)
        workers[worker_type] = procs
    logger.info(f"Node {args.node_rank}: Submitted all workers.")

    # Configure workers.
    for worker_type, spec in system.RL_WORKERS.items():
        if worker_type not in workers:
            continue
        worker_setups = getattr(setup, spec.config_field_name)
        for worker_setup in worker_setups:
            assert worker_setup.worker_info.node_rank == args.node_rank
            worker_setup.worker_info.set_name_resolve_address(
                name_resolve_address, name_resolve_port)

        group_request("configure", [(w, ) for w in worker_setups],
                      worker_tx[worker_type])
    logger.info(f"Node {args.node_rank}: Configured all workers.")

    # Start workers.
    for worker_type in system.RL_WORKERS:
        if worker_type not in workers:
            continue
        group_request("start", [() for _ in worker_tx[worker_type]],
                      worker_tx[worker_type])
    logger.info(
        f"Node {args.node_rank}: Experiment successfully started. Check wandb for progress."
    )

    for w in itertools.chain.from_iterable(workers.values()):
        w.join(timeout=args.timeout)

    for ctrl in itertools.chain.from_iterable(ctrls.values()):
        if ctrl.inf_ctrls is not None:
            for c in ctrl.inf_ctrls:
                c.request_ctrl.close()
                c.response_ctrl.close()
        if ctrl.spl_ctrls is not None:
            for c in ctrl.spl_ctrls:
                c.close()


def main():
    parser = argparse.ArgumentParser(prog="srl-local")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("run", help="starts a basic experiment")
    subparser.add_argument("--node_rank",
                           "-n",
                           type=int,
                           required=False,
                           default=0,
                           help="Rank of the node. 0 = master node.")
    subparser.add_argument("--experiment_name",
                           "-e",
                           type=str,
                           required=True,
                           help="name of the experiment")
    subparser.add_argument("--trial_name",
                           "-f",
                           type=str,
                           required=True,
                           help="name of the trial")
    subparser.add_argument("--wandb_mode",
                           type=str,
                           default="disabled",
                           choices=["online", "offline", "disabled"])
    subparser.add_argument("--timeout",
                           "-t",
                           type=int,
                           default=3600,
                           help="Timeout for the experiment. (seconds)")

    subparser.set_defaults(func=run_local)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
